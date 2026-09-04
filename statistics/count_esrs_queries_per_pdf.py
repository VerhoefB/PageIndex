import argparse
import json
from collections import Counter
from pathlib import Path


def is_selected_for_query(row: dict) -> bool:
    """
    Returns True if the chunk was selected for query generation.
    """
    if row.get("generate_query") is True:
        return True

    if row.get("generate_queries") is True:
        return True

    # Extra fallback: if your file stores generated query text directly
    if row.get("query"):
        return True

    return False


def is_eligible_for_query_generation(row: dict) -> bool:
    """
    Eligible chunks either received queries, or were skipped only because
    they were not selected after random sampling/allocation.
    """

    if row.get("generate_query") is True:
        return True

    if row.get("generate_queries") is True:
        return True

    if row.get("num_queries", 0):
        try:
            if int(row.get("num_queries", 0)) > 0:
                return True
        except Exception:
            pass

    reason = (
        row.get("skip_query_reason")
        or row.get("query_skip_reason")
        or ""
    )

    # In query-plan script:
    # eligible + selected     -> skip_query_reason = ""
    # eligible + not selected -> skip_query_reason = "not_selected"
    if reason in ["", "not_selected"]:
        return True

    return False


def get_doc_name(row: dict) -> str:
    """
    Prefer bank_name for ESRS, otherwise fall back to doc_name/doc_id.
    """
    return (
        row.get("bank_name")
        or row.get("doc_name")
        or row.get("doc_id")
        or "UNKNOWN"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Count ESRS query-generation chunks per PDF/bank."
    )

    parser.add_argument(
        "--chunks-with-query-plan",
        required=True,
        help="Path to ESRS_combined_chunks_with_query_plan.jsonl"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file with the summary"
    )

    args = parser.parse_args()

    input_path = Path(args.chunks_with_query_plan)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    total_chunks = 0
    selected_query_chunks = 0

    chunks_per_pdf = Counter()
    selected_queries_per_pdf = Counter()

    skipped_reason_counts = Counter()
    skipped_reason_per_pdf = {}

    eligible_query_chunks = 0
    eligible_queries_per_pdf = Counter()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            total_chunks += 1

            doc_name = get_doc_name(row)
            chunks_per_pdf[doc_name] += 1

            if is_eligible_for_query_generation(row):
                eligible_query_chunks += 1
                eligible_queries_per_pdf[doc_name] += 1

            if is_selected_for_query(row):
                selected_query_chunks += 1
                selected_queries_per_pdf[doc_name] += 1
            else:
                reason = (
                    row.get("skip_query_reason")
                    or row.get("query_skip_reason")
                    or "not_selected_or_no_reason"
                )

                skipped_reason_counts[reason] += 1

                if doc_name not in skipped_reason_per_pdf:
                    skipped_reason_per_pdf[doc_name] = Counter()

                skipped_reason_per_pdf[doc_name][reason] += 1

    skipped_reason_per_pdf_clean = {
        doc_name: dict(counts)
        for doc_name, counts in sorted(skipped_reason_per_pdf.items())
    }

    summary = {
        "input_file": str(input_path),
        "total_chunks": total_chunks,
        "selected_query_chunks": selected_query_chunks,
        "chunks_per_pdf": dict(sorted(chunks_per_pdf.items())),
        "selected_queries_per_pdf": dict(sorted(selected_queries_per_pdf.items())),
        "skipped_reason_counts": dict(sorted(skipped_reason_counts.items())),
        "skipped_reason_per_pdf": skipped_reason_per_pdf_clean,
        "eligible_query_chunks": eligible_query_chunks,
        "eligible_queries_per_pdf": dict(sorted(eligible_queries_per_pdf.items())),
    }

    print("\n=== ESRS QUERY PLAN SUMMARY ===")
    print(f"Input file: {input_path}")
    print(f"Total chunks: {total_chunks}")
    print(f"Eligible query chunks: {eligible_query_chunks}")
    print(f"Selected query chunks: {selected_query_chunks}")

    print("\nEligible query chunks per PDF/bank:")
    for doc_name, count in sorted(eligible_queries_per_pdf.items()):
        print(f"  {doc_name}: {count}")
        
    print("\nSelected queries per PDF/bank:")
    for doc_name, count in sorted(selected_queries_per_pdf.items()):
        print(f"  {doc_name}: {count}")

    print("\nTotal chunks per PDF/bank:")
    for doc_name, count in sorted(chunks_per_pdf.items()):
        print(f"  {doc_name}: {count}")

    print("\nSkipped query reasons:")
    for reason, count in sorted(skipped_reason_counts.items()):
        print(f"  {reason}: {count}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()