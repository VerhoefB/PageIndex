import argparse
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_text(row: dict) -> str:
    return (
        row.get("text")
        or row.get("page_content")
        or row.get("chunk_text")
        or row.get("content")
        or ""
    )


def get_doc_name(row: dict, fallback_name: str) -> str:
    return (
        row.get("bank_name")
        or row.get("doc_name")
        or row.get("doc_id")
        or row.get("document_name")
        or fallback_name
    )


def get_token_count(row: dict):
    value = (
        row.get("token_count")
        or row.get("text_token_count")
        or row.get("num_tokens")
        or row.get("tokens")
    )

    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return stats.stdev(values)


def percentile(values, p):
    """
    Simple percentile using sorted values.
    p should be between 0 and 100.
    """
    if not values:
        return 0

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * (p / 100)
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)
    weight = k - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def summarize_values(values: list[int]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "min": 0,
            "q1": 0,
            "q3": 0,
            "p95": 0,
            "max": 0,
        }

    return {
        "count": len(values),
        "mean": round(stats.mean(values), 2),
        "median": round(stats.median(values), 2),
        "std": round(safe_std(values), 2),
        "min": min(values),
        "q1": round(percentile(values, 25), 2),
        "q3": round(percentile(values, 75), 2),
        "p95": round(percentile(values, 95), 2),
        "max": max(values),
    }


def collect_chunk_files(input_path: Path):
    """
    Supports:
    - one JSONL file
    - one directory containing many *.jsonl files
    """

    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in directory: {input_path}")
        return files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def summarize_dataset(dataset_name: str, input_path: Path) -> dict:
    files = collect_chunk_files(input_path)

    char_lengths_by_doc = defaultdict(list)
    word_lengths_by_doc = defaultdict(list)
    token_lengths_by_doc = defaultdict(list)

    total_rows = 0

    for file_path in files:
        rows = read_jsonl(file_path)
        fallback_doc_name = file_path.stem.replace("_chunks", "")

        for row in rows:
            total_rows += 1

            doc_name = get_doc_name(row, fallback_doc_name)
            text = str(get_text(row) or "")

            char_len = len(text)
            word_len = len(text.split())
            token_len = get_token_count(row)

            char_lengths_by_doc[doc_name].append(char_len)
            word_lengths_by_doc[doc_name].append(word_len)

            if token_len is not None:
                token_lengths_by_doc[doc_name].append(token_len)

    per_doc = []

    all_chars = []
    all_words = []
    all_tokens = []

    for doc_name in sorted(char_lengths_by_doc):
        chars = char_lengths_by_doc[doc_name]
        words = word_lengths_by_doc[doc_name]
        tokens = token_lengths_by_doc.get(doc_name, [])

        all_chars.extend(chars)
        all_words.extend(words)
        all_tokens.extend(tokens)

        char_stats = summarize_values(chars)
        word_stats = summarize_values(words)
        token_stats = summarize_values(tokens)

        per_doc.append({
            "dataset": dataset_name,
            "doc_name": doc_name,

            "number_of_chunks": char_stats["count"],

            "characters_mean": char_stats["mean"],
            "characters_median": char_stats["median"],
            "characters_std": char_stats["std"],
            "characters_min": char_stats["min"],
            "characters_q1": char_stats["q1"],
            "characters_q3": char_stats["q3"],
            "characters_p95": char_stats["p95"],
            "characters_max": char_stats["max"],

            "words_mean": word_stats["mean"],
            "words_median": word_stats["median"],
            "words_std": word_stats["std"],
            "words_min": word_stats["min"],
            "words_q1": word_stats["q1"],
            "words_q3": word_stats["q3"],
            "words_p95": word_stats["p95"],
            "words_max": word_stats["max"],

            "tokens_available": len(tokens) > 0,
            "tokens_mean": token_stats["mean"],
            "tokens_median": token_stats["median"],
            "tokens_std": token_stats["std"],
            "tokens_min": token_stats["min"],
            "tokens_q1": token_stats["q1"],
            "tokens_q3": token_stats["q3"],
            "tokens_p95": token_stats["p95"],
            "tokens_max": token_stats["max"],
        })

    total_char_stats = summarize_values(all_chars)
    total_word_stats = summarize_values(all_words)
    total_token_stats = summarize_values(all_tokens)

    total_row = {
        "dataset": dataset_name,
        "doc_name": "Total",

        "number_of_chunks": total_char_stats["count"],

        "characters_mean": total_char_stats["mean"],
        "characters_median": total_char_stats["median"],
        "characters_std": total_char_stats["std"],
        "characters_min": total_char_stats["min"],
        "characters_q1": total_char_stats["q1"],
        "characters_q3": total_char_stats["q3"],
        "characters_p95": total_char_stats["p95"],
        "characters_max": total_char_stats["max"],

        "words_mean": total_word_stats["mean"],
        "words_median": total_word_stats["median"],
        "words_std": total_word_stats["std"],
        "words_min": total_word_stats["min"],
        "words_q1": total_word_stats["q1"],
        "words_q3": total_word_stats["q3"],
        "words_p95": total_word_stats["p95"],
        "words_max": total_word_stats["max"],

        "tokens_available": len(all_tokens) > 0,
        "tokens_mean": total_token_stats["mean"],
        "tokens_median": total_token_stats["median"],
        "tokens_std": total_token_stats["std"],
        "tokens_min": total_token_stats["min"],
        "tokens_q1": total_token_stats["q1"],
        "tokens_q3": total_token_stats["q3"],
        "tokens_p95": total_token_stats["p95"],
        "tokens_max": total_token_stats["max"],
    }

    return {
        "dataset": dataset_name,
        "input_path": str(input_path),
        "num_files": len(files),
        "num_documents": len(char_lengths_by_doc),
        "total_chunks": total_rows,
        "per_doc": per_doc,
        "total": total_row,
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_compact_table(dataset_summary: dict):
    print(f"\n=== {dataset_summary['dataset']} CHUNK STATISTICS ===")
    print(f"Input path: {dataset_summary['input_path']}")
    print(f"Files read: {dataset_summary['num_files']}")
    print(f"Documents: {dataset_summary['num_documents']}")
    print(f"Total chunks: {dataset_summary['total_chunks']}")

    print("\nDocName | NumberOfChunks | MedianChars | StdChars | MaxChars")
    print("-" * 75)

    for row in dataset_summary["per_doc"]:
        print(
            f"{row['doc_name']} | "
            f"{row['number_of_chunks']} | "
            f"{row['characters_median']} | "
            f"{row['characters_std']} | "
            f"{row['characters_max']}"
        )

    total = dataset_summary["total"]
    print("-" * 75)
    print(
        f"Total | "
        f"{total['number_of_chunks']} | "
        f"{total['characters_median']} | "
        f"{total['characters_std']} | "
        f"{total['characters_max']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize chunk length statistics for ESRS and FinanceBench."
    )

    parser.add_argument(
        "--esrs-chunks",
        default=None,
        help="Path to ESRS chunks JSONL file or directory with ESRS chunk JSONL files."
    )

    parser.add_argument(
        "--financebench-chunks",
        default=None,
        help="Path to FinanceBench chunks JSONL file or directory with FinanceBench chunk JSONL files."
    )

    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON file with full statistics."
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV file with thesis/table-friendly statistics."
    )

    args = parser.parse_args()

    summaries = []
    csv_rows = []

    if args.esrs_chunks:
        esrs_summary = summarize_dataset(
            dataset_name="ESRS",
            input_path=Path(args.esrs_chunks)
        )
        summaries.append(esrs_summary)
        csv_rows.extend(esrs_summary["per_doc"])
        csv_rows.append(esrs_summary["total"])
        print_compact_table(esrs_summary)

    if args.financebench_chunks:
        financebench_summary = summarize_dataset(
            dataset_name="FinanceBench",
            input_path=Path(args.financebench_chunks)
        )
        summaries.append(financebench_summary)
        csv_rows.extend(financebench_summary["per_doc"])
        csv_rows.append(financebench_summary["total"])
        print_compact_table(financebench_summary)

    if not summaries:
        raise ValueError("Provide at least one of --esrs-chunks or --financebench-chunks.")

    output = {
        "datasets": summaries
    }

    write_json(Path(args.output_json), output)
    write_csv(Path(args.output_csv), csv_rows)

    print(f"\nSaved JSON to: {args.output_json}")
    print(f"Saved CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()