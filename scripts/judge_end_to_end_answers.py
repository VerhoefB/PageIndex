import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)

JUDGE_PROMPT_VERSION = "judge_v1"
JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "category": {
            "type": "string",
            "enum": ["correct", "partially_correct", "incorrect", "no_answer"],
        },
        "rationale": {"type": "string"},
    },
    "required": ["category", "rationale"],
}
CATEGORIES = ["correct", "partially_correct", "incorrect", "no_answer"]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


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


def judge_prompt(query: str, ground_truth_answer: str, generated_answer: str) -> str:
    return f"""
Judge the generated answer for a retrieval-augmented QA experiment.

You receive only:
- the question,
- the ground-truth/reference answer,
- the generated answer.

You do not receive the retrieval method or retrieved chunks. Evaluate factual correctness and completeness relative to the reference answer and the question.

Use exactly one category:
- correct: The generated answer is factually correct and sufficiently complete to answer the question.
- partially_correct: The core answer is correct, but one or more important elements required by the reference answer are missing, incomplete, or slightly inaccurate.
- incorrect: The generated answer is factually wrong, misleading, contradicts the reference answer, or answers a different question.
- no_answer: The generated answer explicitly states that there is insufficient information to answer the question.

Do not penalize harmless wording differences, formatting differences, or equivalent numerical expressions.

Question:
{query}

Ground-truth/reference answer:
{ground_truth_answer}

Generated answer:
{generated_answer}

Return only the structured judgement.
""".strip()


def call_judge(
    client: OpenAI,
    model: str,
    query: str,
    ground_truth_answer: str,
    generated_answer: str,
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
                        "content": "You are a strict, consistent, and method-blind LLM-as-a-judge evaluator.",
                    },
                    {
                        "role": "user",
                        "content": judge_prompt(query, ground_truth_answer, generated_answer),
                    },
                ],
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "end_to_end_judgement",
                        "schema": JUDGE_SCHEMA,
                        "strict": True,
                    }
                },
            )
            latency = time.perf_counter() - start
            judgement = json.loads(response.output_text)
            return {
                "judge_success": True,
                "judge_error": "",
                "judge_category": judgement["category"],
                "judge_rationale": judgement["rationale"],
                "judge_usage": usage_to_dict(response),
                "judge_latency_seconds": round(latency, 6),
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
        "judge_success": False,
        "judge_error": last_error or "Unknown judge error",
        "judge_category": "",
        "judge_rationale": "",
        "judge_usage": None,
        "judge_latency_seconds": None,
        "stop_run": bool(
            last_exception
            and (is_rate_limit_error(last_exception) or is_quota_or_billing_error(last_exception))
        ),
    }


def latest_successful_generated_answers(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Return one successful generation row per answer_id.

    This safely ignores earlier failed attempts when a later resume succeeded.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            answer_id = row.get("answer_id")
            if not answer_id:
                continue
            if row.get("generation_success"):
                latest[str(answer_id)] = row
    return latest


def load_successful_judge_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    completed: Set[str] = set()
    for row in read_jsonl(output_path):
        if row.get("judge_success") and row.get("judge_id"):
            completed.add(str(row["judge_id"]))
    return completed


def latest_successful_judgements(output_path: Path) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    if not output_path.exists():
        return latest
    for row in read_jsonl(output_path):
        judge_id = row.get("judge_id")
        if judge_id and row.get("judge_success"):
            latest[str(judge_id)] = row
    return latest


def write_summary_csv(output_path: Path, summary_path: Path) -> None:
    rows = list(latest_successful_judgements(output_path).values())
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (
            row.get("dataset"),
            row.get("configuration_id"),
            row.get("method"),
            row.get("retrieval_model"),
            row.get("variant"),
            row.get("top_m"),
            row.get("retrieval_setting"),
        )
        grouped[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        counts = defaultdict(int)
        for row in group:
            counts[row.get("judge_category", "")] += 1

        total = len(group)
        summary: Dict[str, Any] = {
            "dataset": key[0],
            "configuration_id": key[1],
            "method": key[2],
            "retrieval_model": key[3],
            "variant": key[4],
            "top_m": key[5],
            "retrieval_setting": key[6],
            "n_judged": total,
        }
        for category in CATEGORIES:
            count = counts[category]
            summary[f"{category}_count"] = count
            summary[f"{category}_pct"] = round(100.0 * count / total, 3) if total else 0.0

        summary_rows.append(summary)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Judge previously generated RAG answers. Writes every judgement immediately, "
            "resumes from successful judgements, and produces a partial/final summary CSV."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Generated-answer JSONL files.")
    parser.add_argument("--output", required=True, help="Judgement JSONL output path.")
    parser.add_argument("--summary-csv", required=True, help="Aggregate category summary CSV.")
    parser.add_argument("--model", default="gpt-5", help="Judge model.")
    parser.add_argument("--max-output-tokens", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional generated-answer limit for testing.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip successful judge_ids already present in the output.",
    )
    args = parser.parse_args()

    input_paths = [Path(x) for x in args.inputs]
    output_path = Path(args.output)
    summary_path = Path(args.summary_csv)

    generated = latest_successful_generated_answers(input_paths)
    completed = set() if args.no_resume else load_successful_judge_ids(output_path)

    print(f"Successful generated answers available: {len(generated)}")
    print(f"Already judged successfully: {len(completed)}")
    print(f"Judge model: {args.model}")
    print(f"Output: {output_path}")

    processed_this_run = 0

    try:
        for answer_id, answer_row in generated.items():
            if args.max_rows is not None and processed_this_run >= args.max_rows:
                print("Reached --max-rows.")
                break

            judge_id = answer_id
            if judge_id in completed:
                continue

            query = str(answer_row.get("query") or "").strip()
            ground_truth_answer = str(answer_row.get("ground_truth_answer") or "").strip()
            generated_answer = str(answer_row.get("generated_answer") or "").strip()

            if not query or not ground_truth_answer or not generated_answer:
                print(f"Skipping {answer_id}: missing query/reference/generated answer.")
                continue

            NO_ANSWER_TEXT = "Insufficient information to answer the question."

            if generated_answer.strip().lower() == NO_ANSWER_TEXT.lower():
                result = {
                    "judge_success": True,
                    "judge_error": "",
                    "fatal_quota_error": False,
                    "judge_category": "no_answer",
                    "judge_rationale": (
                        "The answer-generation model explicitly stated that the "
                        "retrieved context contained insufficient information."
                    ),
                    "judge_usage": None,
                    "judge_response_id": None,
                }
            else:
                result = call_judge(
                    client=client,
                    model=args.model,
                    query=query,
                    ground_truth_answer=ground_truth_answer,
                    generated_answer=generated_answer,
                    max_output_tokens=args.max_output_tokens,
                    max_retries=args.max_retries,
                    sleep_seconds=args.sleep_seconds,
                )

            output_row: Dict[str, Any] = {
                "judge_id": judge_id,
                "answer_id": answer_id,
                "dataset": answer_row.get("dataset"),
                "source_result_file": answer_row.get("source_result_file"),
                "configuration_id": answer_row.get("configuration_id"),
                "method": answer_row.get("method"),
                "retrieval_model": answer_row.get("retrieval_model"),
                "variant": answer_row.get("variant"),
                "top_m": answer_row.get("top_m"),
                "query_id": answer_row.get("query_id"),
                "financebench_id": answer_row.get("financebench_id"),
                "company": answer_row.get("company"),
                "doc_name": answer_row.get("doc_name"),
                "bank_name": answer_row.get("bank_name"),
                "retrieval_setting": answer_row.get("retrieval_setting"),
                "top_k": answer_row.get("top_k"),
                "query": query,
                "ground_truth_answer": ground_truth_answer,
                "generated_answer": generated_answer,
                "generation_model": answer_row.get("generation_model"),
                "generation_prompt_version": answer_row.get("generation_prompt_version"),
                "judge_model": args.model,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                **result,
                "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            append_jsonl_durable(output_path, output_row)
            processed_this_run += 1

            if result["judge_success"]:
                completed.add(judge_id)
                print(
                    f"[{processed_this_run}] {answer_row.get('configuration_id')} | "
                    f"{answer_row.get('query_id')} | {answer_row.get('retrieval_setting')} | "
                    f"{result['judge_category']}"
                )
            else:
                print(
                    f"[{processed_this_run}] {answer_id} | FAILED: {result['judge_error']}"
                )

            if result.get("stop_run"):
                print(
                    "Stopping after a rate-limit/quota-style error. Prior judgements are "
                    "already on disk. Re-run the same command to resume."
                )
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Completed judgements are already saved; rerun to resume.")
    finally:
        write_summary_csv(output_path, summary_path)
        print("\n=== JUDGE STATUS ===")
        print(f"Judgement attempts appended this run: {processed_this_run}")
        print(f"Successful judge_ids currently known: {len(completed)}")
        print(f"Judgement JSONL: {output_path}")
        print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()