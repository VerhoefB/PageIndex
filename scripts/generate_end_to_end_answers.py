import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)

GENERATION_PROMPT_VERSION = "generation_v1"
INSUFFICIENT_ANSWER = "Insufficient information to answer the question."


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc
    return rows


def append_jsonl_durable(path: Path, row: Dict[str, Any]) -> None:
    """Append one JSONL row and force it to disk immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def first_existing(row: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return default


def stable_query_id(row: Dict[str, Any]) -> str:
    """Create the same query identifier across retrieval methods."""
    financebench_id = row.get("financebench_id")
    if financebench_id not in (None, ""):
        return str(financebench_id)

    query_id = row.get("query_id")
    if query_id not in (None, ""):
        # For ESRS, prefer the ground-truth chunk ID across retrieval methods.
        gt = row.get("ground_truth_chunk_id")
        if gt not in (None, ""):
            return str(gt)
        return str(query_id)

    gt = row.get("ground_truth_chunk_id")
    if gt not in (None, ""):
        return str(gt)

    query = str(first_existing(row, ["query", "question", "user_question"], ""))
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
    return f"query_{digest}"


def configuration_id(input_path: Path) -> str:
    """Use the retrieval result filename as the unique configuration identifier."""
    stem = input_path.stem
    if stem.endswith("_query_results"):
        stem = stem[: -len("_query_results")]
    return stem


def infer_variant(row: Dict[str, Any], input_path: Path) -> str:
    mode = str(row.get("mode") or "").strip()
    name = input_path.name.lower()
    method = str(row.get("method") or "").lower()

    if "chunk_rerank" in name or "chunk_rerank" in method or mode == "chunk_rerank":
        return "chunk_rerank"
    if "hybrid_pageindex" in name or "hybrid_pageindex" in method:
        return "node"
    if "pageindex" in name or "pageindex" in method:
        return "standard"
    return mode or "standard"


def extract_chunk_text(chunk: Any) -> str:
    """Extract only the chunk text used for answer generation."""
    if isinstance(chunk, str):
        return chunk.strip()
    if isinstance(chunk, dict):
        for key in ("text", "chunk_text", "page_content", "content", "node_text"):
            value = chunk.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def extract_chunks_for_setting(row: Dict[str, Any], top_k: int) -> List[Any]:
    """
    Use the stored top-1 or top-5 retrieval output.

    For PageIndex, top-1 and top-5 are separate outputs from the same
    traversal, so top-1 must not be taken as top5_chunks[:1].
    """
    if top_k == 1:
        candidates = row.get("top1_chunks")
        if isinstance(candidates, list):
            return candidates[:1]

        # Compatibility fallback for older files only.
        for key in ("retrieved_chunks", "top_chunks", "results", "retrieved_results"):
            candidates = row.get(key)
            if isinstance(candidates, list):
                return candidates[:1]
        return []

    if top_k == 5:
        candidates = row.get("top5_chunks")
        if isinstance(candidates, list):
            return candidates[:5]

        # Compatibility fallback for older files only.
        for key in ("retrieved_chunks", "top_chunks", "results", "retrieved_results", "top_5"):
            candidates = row.get(key)
            if isinstance(candidates, list):
                return candidates[:5]
        return []

    raise ValueError("This thesis evaluation supports top_k settings 1 and 5 only.")


def extract_chunk_ids_for_setting(row: Dict[str, Any], chunks: List[Any], top_k: int) -> List[str]:
    key = "top1_chunk_ids" if top_k == 1 else "top5_chunk_ids"
    ids = row.get(key)
    if isinstance(ids, list):
        return [str(x) for x in ids[:top_k]]

    output: List[str] = []
    for chunk in chunks[:top_k]:
        if isinstance(chunk, dict):
            chunk_id = first_existing(
                chunk,
                ["chunk_id", "retrieval_chunk_id", "canonical_chunk_id", "id", "node_id"],
                None,
            )
            if chunk_id is not None:
                output.append(str(chunk_id))
    return output


def format_context(chunks: List[Any], max_chars_per_chunk: int) -> Tuple[str, List[int]]:
    parts: List[str] = []
    original_lengths: List[int] = []

    for rank, chunk in enumerate(chunks, start=1):
        text = extract_chunk_text(chunk)
        if not text:
            continue

        original_lengths.append(len(text))
        if max_chars_per_chunk > 0 and len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "\n[TRUNCATED BY EVALUATION SCRIPT]"

        parts.append(f"[Retrieved chunk {rank}]\n{text}")

    if not parts:
        return "No retrieved chunk text was available.", original_lengths

    return "\n\n".join(parts), original_lengths


def generation_prompt(query: str, context: str) -> str:
    return f"""
Answer the question using only the retrieved context.

Rules:
- Use only information contained in the retrieved context.
- Do not use outside knowledge.
- Do not invent, assume, or add unsupported facts.
- Answer the question directly and concisely, while including all information needed for a complete answer.
- If the retrieved context does not contain enough information to answer the question, output exactly:
  {INSUFFICIENT_ANSWER}

Question:
{query}

Retrieved context:
{context}
""".strip()


def usage_to_dict(response: Any) -> Optional[Dict[str, Any]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    try:
        return dict(usage)
    except Exception:
        return {"raw": str(usage)}


def is_quota_or_billing_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "insufficient_quota" in text
        or "billing" in text
        or "exceeded your current quota" in text
        or "usage limit" in text
    )


def is_rate_limit_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    return "ratelimit" in name or "rate limit" in text or "too many requests" in text


def call_generator(
    client: OpenAI,
    model: str,
    query: str,
    context: str,
    max_output_tokens: int,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    last_error: Optional[str] = None
    last_exception: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            start = time.perf_counter()
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are the answer-generation component of a controlled "
                            "retrieval-augmented generation experiment."
                        ),
                    },
                    {"role": "user", "content": generation_prompt(query, context)},
                ],
                max_output_tokens=max_output_tokens,
            )
            latency = time.perf_counter() - start
            return {
                "generation_success": True,
                "generation_error": "",
                "generated_answer": response.output_text.strip(),
                "generation_usage": usage_to_dict(response),
                "generation_latency_seconds": round(latency, 6),
                "stop_run": False,
            }
        except Exception as exc:
            last_exception = exc
            last_error = str(exc)

            # Quota/billing errors require user action, so do not burn retries.
            if is_quota_or_billing_error(exc):
                break

            if attempt < max_retries:
                time.sleep(sleep_seconds * (2 ** (attempt - 1)))

    return {
        "generation_success": False,
        "generation_error": last_error or "Unknown generation error",
        "generated_answer": "",
        "generation_usage": None,
        "generation_latency_seconds": None,
        "stop_run": bool(
            last_exception
            and (is_rate_limit_error(last_exception) or is_quota_or_billing_error(last_exception))
        ),
    }


def load_successful_answer_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    successful: Set[str] = set()
    for row in read_jsonl(output_path):
        if (
            row.get("generation_success")
            and row.get("answer_id")
            and str(row.get("generated_answer", "")).strip()
        ):
            successful.add(str(row["answer_id"]))
    return successful


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate end-to-end RAG answers from existing retrieval JSONL files. "
            "Writes every API result immediately and resumes from successful rows."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Retrieval result JSONL files.")
    parser.add_argument("--dataset", required=True, choices=["ESRS", "FinanceBench"])
    parser.add_argument("--output", required=True, help="Generated-answer JSONL output path.")
    parser.add_argument("--model", default="gpt-5", help="Answer generation model.")
    parser.add_argument(
        "--top-k-settings",
        nargs="+",
        type=int,
        default=[1, 5],
        choices=[1, 5],
        help="Normally evaluate both top-1 and top-5.",
    )
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=0,
        help=(
            "0 = no truncation (recommended for the thesis). Set a positive value only "
            "if you intentionally want to cap each retrieved chunk."
        ),
    )
    parser.add_argument("--max-output-tokens", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional query-row limit for testing.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip successful answer_ids already present in the output.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    completed = set() if args.no_resume else load_successful_answer_ids(output_path)

    print(f"Dataset: {args.dataset}")
    print(f"Generation model: {args.model}")
    print(f"Already completed answer rows: {len(completed)}")
    print(f"Output: {output_path}")

    total_query_rows_seen = 0
    total_api_rows_written = 0

    try:
        for input_name in args.inputs:
            input_path = Path(input_name)
            config = configuration_id(input_path)
            rows = read_jsonl(input_path)

            print(f"\n--- {config} ({len(rows)} retrieval rows) ---")

            for row_index, row in enumerate(rows):
                if args.max_rows is not None and total_query_rows_seen >= args.max_rows:
                    print("Reached --max-rows.")
                    return

                total_query_rows_seen += 1
                query = str(first_existing(row, ["query", "question", "user_question"], "")).strip()
                ground_truth_answer = str(
                    first_existing(
                        row,
                        ["ground_truth_answer", "gold_answer", "reference_answer", "answer", "expected_answer"],
                        "",
                    )
                ).strip()

                if not query:
                    print(f"Skipping {config} row {row_index}: no query.")
                    continue
                if not ground_truth_answer:
                    print(f"Skipping {config} row {row_index}: no ground-truth answer.")
                    continue

                query_id = stable_query_id(row)

                for top_k in args.top_k_settings:
                    answer_id = f"{args.dataset}::{config}::{query_id}::top_{top_k}"
                    if answer_id in completed:
                        continue

                    chunks = extract_chunks_for_setting(row, top_k)
                    chunk_ids = extract_chunk_ids_for_setting(row, chunks, top_k)
                    context, original_lengths = format_context(chunks, args.max_chars_per_chunk)

                    result = call_generator(
                        client=client,
                        model=args.model,
                        query=query,
                        context=context,
                        max_output_tokens=args.max_output_tokens,
                        max_retries=args.max_retries,
                        sleep_seconds=args.sleep_seconds,
                    )

                    output_row: Dict[str, Any] = {
                        "answer_id": answer_id,
                        "dataset": args.dataset,
                        "source_result_file": input_path.name,
                        "configuration_id": config,
                        "method": row.get("method"),
                        "retrieval_model": row.get("model"),
                        "variant": infer_variant(row, input_path),
                        "top_m": row.get("top_m"),
                        "query_id": query_id,
                        "financebench_id": row.get("financebench_id"),
                        "company": row.get("company"),
                        "doc_name": row.get("doc_name") or row.get("bank_name"),
                        "bank_name": row.get("bank_name", ""),
                        "query": query,
                        "ground_truth_answer": ground_truth_answer,
                        "ground_truth_chunk_id": row.get("ground_truth_chunk_id"),
                        "retrieval_setting": f"top_{top_k}",
                        "top_k": top_k,
                        "retrieved_chunk_ids": chunk_ids,
                        "num_retrieved_chunks": len(chunks),
                        "original_chunk_character_lengths": original_lengths,
                        "max_chars_per_chunk": args.max_chars_per_chunk,
                        "generation_model": args.model,
                        "generation_prompt_version": GENERATION_PROMPT_VERSION,
                        **result,
                        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    append_jsonl_durable(output_path, output_row)
                    total_api_rows_written += 1

                    if result["generation_success"]:
                        completed.add(answer_id)
                        print(
                            f"[{total_api_rows_written}] {config} | {query_id} | top_{top_k} | generated"
                        )
                    else:
                        print(
                            f"[{total_api_rows_written}] {config} | {query_id} | top_{top_k} | "
                            f"FAILED: {result['generation_error']}"
                        )

                    if result.get("stop_run"):
                        print(
                            "Stopping after a rate-limit/quota-style error. All successful prior "
                            "answers are already on disk. Re-run the same command to resume."
                        )
                        return
    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed answers are already saved; rerun to resume.")
        return
    finally:
        print("\n=== ANSWER GENERATION STATUS ===")
        print(f"Retrieval query rows seen this run: {total_query_rows_seen}")
        print(f"API result rows appended this run: {total_api_rows_written}")
        print(f"Successful answer_ids currently known: {len(completed)}")
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()