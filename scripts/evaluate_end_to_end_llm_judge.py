import argparse
import csv
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

from openai import OpenAI


JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "category": {
            "type": "string",
            "enum": ["correct", "partially_correct", "incorrect", "no_answer"],
        },
        "rationale": {
            "type": "string",
        },
    },
    "required": ["category", "rationale"],
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}: {exc}") from exc

    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_first_existing(row: Dict[str, Any], keys: List[str], default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)

        if value is not None and value != "":
            return value

    return default


def extract_query(row: Dict[str, Any]) -> str:
    return get_first_existing(row, ["query", "question", "user_question"])


def extract_ground_truth_answer(row: Dict[str, Any]) -> str:
    return get_first_existing(
        row,
        [
            "ground_truth_answer",
            "gold_answer",
            "reference_answer",
            "answer",
            "expected_answer",
        ],
    )


def extract_query_id(row: Dict[str, Any], index: int) -> str:
    query_id = get_first_existing(row, ["query_id", "question_id", "id"], None)

    if query_id is not None:
        return str(query_id)

    return f"row_{index}"


def extract_method(row: Dict[str, Any], input_path: Path) -> str:
    method = get_first_existing(
        row,
        ["method", "retriever", "retrieval_method", "model_name", "system"],
        None,
    )

    if method is not None:
        return str(method)

    return input_path.stem


def extract_chunk_text(chunk: Any) -> str:
    """
    Extract only the raw text of the retrieved chunk.

    Important:
    Do NOT pass the full chunk dictionary to the generation model.
    The chunk dictionary may contain evaluation metadata such as
    ground-truth labels, ranks, scores, or correctness indicators.
    """

    if isinstance(chunk, str):
        return chunk

    if isinstance(chunk, dict):
        for key in [
            "text",
            "chunk_text",
            "page_content",
            "content",
            "node_text",
        ]:
            value = chunk.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def extract_retrieved_chunks(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tries to support common retrieval-output schemas.

    Expected possibilities:
    - retrieved_chunks: [{...}, {...}]
    - top_chunks: [{...}, {...}]
    - results: [{...}, {...}]
    - retrieved_results: [{...}, {...}]
    - top_5: [{...}, {...}]
    """

    candidates = get_first_existing(
        row,
        [
            "retrieved_chunks",
            "top_chunks",
            "retrieved_results",
            "results",
            "top_5",
            "top_k_results",
        ],
        [],
    )

    if candidates is None:
        return []

    if not isinstance(candidates, list):
        return []

    output = []

    for rank, chunk in enumerate(candidates, start=1):
        if isinstance(chunk, dict):
            chunk_copy = dict(chunk)
            chunk_copy["rank"] = chunk_copy.get("rank", rank)
            chunk_copy["text_for_generation"] = extract_chunk_text(chunk)
            output.append(chunk_copy)
        else:
            output.append(
                {
                    "rank": rank,
                    "text_for_generation": extract_chunk_text(chunk),
                }
            )

    return output


def format_chunks_for_prompt(
    chunks: List[Dict[str, Any]],
    top_k: int,
    max_chars_per_chunk: int,
) -> str:
    selected = chunks[:top_k]

    if not selected:
        return "No retrieved chunks were provided."

    parts = []

    for display_rank, chunk in enumerate(selected, start=1):
        text = chunk.get("text_for_generation", "")

        if not text:
            continue

        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "\n[TRUNCATED]"

        parts.append(f"[Retrieved chunk {display_rank}]\n{text}")

    if not parts:
        return "No retrieved chunk text was available."

    return "\n\n".join(parts)


def generate_answer(
    client: OpenAI,
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
    model: str,
    temperature: float,
    max_chars_per_chunk: int,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    context = format_chunks_for_prompt(
        chunks=chunks,
        top_k=top_k,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    prompt = f"""
You are answering a question using only the retrieved context below.

Rules:
- Answer the question only if the retrieved context contains enough information.
- Do not use outside knowledge.
- Do not invent facts.
- If the retrieved context does not contain enough information, answer exactly:
  Insufficient information to answer the question.

Question:
{query}

Retrieved context:
{context}
""".strip()

    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": "You generate grounded answers for a retrieval-augmented evaluation.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=temperature,
            )

            return {
                "generation_success": True,
                "generation_error": "",
                "generated_answer": response.output_text.strip(),
                "generation_usage": response.usage.model_dump() if response.usage else None,
            }

        except Exception as exc:
            last_error = str(exc)

            if attempt < max_retries:
                time.sleep(sleep_seconds * attempt)

    return {
        "generation_success": False,
        "generation_error": last_error or "Unknown generation error",
        "generated_answer": "",
        "generation_usage": None,
    }


def judge_answer(
    client: OpenAI,
    query: str,
    ground_truth_answer: str,
    generated_answer: str,
    model: str,
    temperature: float,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    prompt = f"""
    You are judging the quality of a generated answer in a retrieval-augmented QA experiment.

    You receive:
    - the question,
    - the ground-truth answer,
    - the generated answer.

    You do NOT receive the retrieval method.
    You do NOT receive the retrieved chunks.
    Judge only based on factual correctness and completeness relative to the ground-truth answer.

    Categories:
    - correct: The generated answer is factually correct and sufficiently complete.
    - partially_correct: The core information is correct, but one or more important elements from the ground-truth answer are missing or inaccurate.
    - incorrect: The answer is factually wrong, misleading, contradicts the ground-truth answer, or answers a different question.
    - no_answer: The generated answer explicitly states that there is insufficient information to answer.

    Question:
    {query}

    Ground-truth answer:
    {ground_truth_answer}

    Generated answer:
    {generated_answer}

    Return only the structured judgement.
    """.strip()

    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": "You are a strict but fair LLM-as-a-judge evaluator.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=temperature,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "end_to_end_judgement",
                        "schema": JUDGE_SCHEMA,
                        "strict": True,
                    }
                },
            )

            judgement = json.loads(response.output_text)

            return {
                "judge_success": True,
                "judge_error": "",
                "judge_category": judgement["category"],
                "judge_rationale": judgement["rationale"],
                "judge_usage": response.usage.model_dump() if response.usage else None,
            }

        except Exception as exc:
            last_error = str(exc)

            if attempt < max_retries:
                time.sleep(sleep_seconds * attempt)

    return {
        "judge_success": False,
        "judge_error": last_error or "Unknown judge error",
        "judge_category": "",
        "judge_rationale": "",
        "judge_usage": None,
    }


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    completed = set()

    for row in read_jsonl(output_path):
        row_id = row.get("end_to_end_row_id")

        if row_id:
            completed.add(row_id)

    return completed


def aggregate_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)

    for row in rows:
        if not row.get("judge_success"):
            continue

        dataset = row.get("dataset", "unknown")
        method = row.get("method", "unknown")
        retrieval_setting = row.get("retrieval_setting", "unknown")

        grouped[(dataset, method, retrieval_setting)].append(row)

    aggregate_rows = []

    categories = ["correct", "partially_correct", "incorrect", "no_answer"]

    for (dataset, method, retrieval_setting), group_rows in sorted(grouped.items()):
        total = len(group_rows)

        output_row = {
            "dataset": dataset,
            "method": method,
            "retrieval_setting": retrieval_setting,
            "num_answers": total,
        }

        counts = defaultdict(int)

        for row in group_rows:
            counts[row["judge_category"]] += 1

        for category in categories:
            output_row[f"{category}_count"] = counts[category]
            output_row[f"{category}_percentage"] = round(100 * counts[category] / total, 2) if total else 0

        aggregate_rows.append(output_row)

    return aggregate_rows


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end RAG evaluation: answer generation from top-1/top-5 chunks and LLM-as-a-judge categorization."
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more retrieval result JSONL files.",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. ESRS or FinanceBench.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file with generated answers and judge categories.",
    )

    parser.add_argument(
        "--aggregate-json",
        required=True,
        help="Output JSON file with category percentages.",
    )

    parser.add_argument(
        "--aggregate-csv",
        required=True,
        help="Output CSV file with category percentages.",
    )

    parser.add_argument(
        "--generation-model",
        default="gpt-5",
        help="Model used for answer generation.",
    )

    parser.add_argument(
        "--judge-model",
        default="gpt-5",
        help="Model used as LLM judge.",
    )

    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=0.0,
        help="Temperature for answer generation.",
    )

    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for judging.",
    )

    parser.add_argument(
        "--top-k-settings",
        nargs="+",
        type=int,
        default=[1, 5],
        help="Retrieval settings to evaluate, usually 1 and 5.",
    )

    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=6000,
        help="Maximum characters per retrieved chunk included in the prompt.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per API call.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Base sleep time between retries.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the output file.",
    )

    args = parser.parse_args()

    client = OpenAI()

    output_path = Path(args.output)
    completed_ids = load_completed_ids(output_path) if args.resume else set()

    total_seen = 0
    total_written = 0

    for input_file in args.inputs:
        input_path = Path(input_file)
        rows = read_jsonl(input_path)

        for index, row in enumerate(rows):
            if args.max_rows is not None and total_seen >= args.max_rows:
                break

            total_seen += 1

            query = extract_query(row)
            ground_truth_answer = extract_ground_truth_answer(row)
            query_id = extract_query_id(row, index)
            method = extract_method(row, input_path)
            chunks = extract_retrieved_chunks(row)

            if not query:
                print(f"Skipping row without query: {input_path} row {index}")
                continue

            if not ground_truth_answer:
                print(f"Skipping row without ground-truth answer: {input_path} row {index}")
                continue

            for top_k in args.top_k_settings:
                retrieval_setting = f"top_{top_k}"

                end_to_end_row_id = (
                    f"{args.dataset}::{method}::{query_id}::{retrieval_setting}"
                )

                if end_to_end_row_id in completed_ids:
                    continue

                generation_result = generate_answer(
                    client=client,
                    query=query,
                    chunks=chunks,
                    top_k=top_k,
                    model=args.generation_model,
                    temperature=args.generation_temperature,
                    max_chars_per_chunk=args.max_chars_per_chunk,
                    max_retries=args.max_retries,
                    sleep_seconds=args.sleep_seconds,
                )

                if generation_result["generation_success"]:
                    judge_result = judge_answer(
                        client=client,
                        query=query,
                        ground_truth_answer=ground_truth_answer,
                        generated_answer=generation_result["generated_answer"],
                        model=args.judge_model,
                        temperature=args.judge_temperature,
                        max_retries=args.max_retries,
                        sleep_seconds=args.sleep_seconds,
                    )
                else:
                    judge_result = {
                        "judge_success": False,
                        "judge_error": "Generation failed, so judging was skipped.",
                        "judge_category": "",
                        "judge_rationale": "",
                        "judge_usage": None,
                    }

                output_row = {
                    "end_to_end_row_id": end_to_end_row_id,
                    "dataset": args.dataset,
                    "method": method,
                    "query_id": query_id,
                    "retrieval_setting": retrieval_setting,
                    "top_k": top_k,
                    "query": query,
                    "ground_truth_answer": ground_truth_answer,
                    "num_retrieved_chunks_available": len(chunks),
                    "retrieved_chunk_ids": [
                        chunk.get("chunk_id") or chunk.get("id") or chunk.get("node_id")
                        for chunk in chunks[:top_k]
                    ],
                    **generation_result,
                    **judge_result,
                }

                append_jsonl(output_path, output_row)
                total_written += 1

                print(
                    f"[{total_seen}] {args.dataset} | {method} | {query_id} | "
                    f"{retrieval_setting} | category={output_row.get('judge_category')}"
                )

        if args.max_rows is not None and total_seen >= args.max_rows:
            break

    judged_rows = read_jsonl(output_path)
    aggregate_rows = aggregate_results(judged_rows)

    write_json(
        Path(args.aggregate_json),
        {
            "dataset": args.dataset,
            "generation_model": args.generation_model,
            "judge_model": args.judge_model,
            "generation_temperature": args.generation_temperature,
            "judge_temperature": args.judge_temperature,
            "top_k_settings": args.top_k_settings,
            "results": aggregate_rows,
        },
    )

    write_csv(Path(args.aggregate_csv), aggregate_rows)

    print("\nDone.")
    print(f"Rows seen: {total_seen}")
    print(f"Rows written: {total_written}")
    print(f"Output JSONL: {output_path}")
    print(f"Aggregate JSON: {args.aggregate_json}")
    print(f"Aggregate CSV: {args.aggregate_csv}")


if __name__ == "__main__":
    main()