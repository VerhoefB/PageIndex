import argparse
import json
from collections import Counter
from pathlib import Path


def load_pdf_doc_names(pdf_folder: Path):
    return {
        pdf_path.stem
        for pdf_path in pdf_folder.glob("*.pdf")
    }


def load_jsonl(path: Path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize FinanceBench questions using both the original "
            "FinanceBench JSONL and the final retained query JSONL."
        )
    )

    parser.add_argument(
        "--pdf-folder",
        required=True,
        help="Folder containing the FinanceBench PDF files, e.g. data/FinanceBench"
    )

    parser.add_argument(
        "--financebench-jsonl",
        required=True,
        help="Original FinanceBench open-source JSONL file"
    )

    parser.add_argument(
        "--queries-jsonl",
        required=True,
        help="Final retained FinanceBench_queries.jsonl file after evidence-to-chunk matching"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file with the summary"
    )

    args = parser.parse_args()

    pdf_folder = Path(args.pdf_folder)
    financebench_jsonl = Path(args.financebench_jsonl)
    queries_jsonl = Path(args.queries_jsonl)

    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    if not financebench_jsonl.exists():
        raise FileNotFoundError(f"FinanceBench JSONL file not found: {financebench_jsonl}")

    if not queries_jsonl.exists():
        raise FileNotFoundError(f"Queries JSONL file not found: {queries_jsonl}")

    pdf_doc_names = load_pdf_doc_names(pdf_folder)

    original_rows = load_jsonl(financebench_jsonl)
    retained_query_rows = load_jsonl(queries_jsonl)

    # Original FinanceBench filtering summary
    total_questions_in_jsonl = 0
    questions_with_available_pdf = 0
    questions_with_single_evidence_and_available_pdf = 0

    skipped_missing_pdf = 0
    skipped_no_evidence = 0
    skipped_multiple_evidence = 0

    missing_doc_names = Counter()
    no_evidence_doc_names = Counter()
    multiple_evidence_doc_names = Counter()

    single_evidence_per_pdf = Counter()

    for row in original_rows:
        total_questions_in_jsonl += 1

        doc_name = row.get("doc_name")
        evidence = row.get("evidence", [])

        if doc_name not in pdf_doc_names:
            skipped_missing_pdf += 1
            missing_doc_names[doc_name] += 1
            continue

        questions_with_available_pdf += 1

        if not isinstance(evidence, list) or len(evidence) == 0:
            skipped_no_evidence += 1
            no_evidence_doc_names[doc_name] += 1
            continue

        if len(evidence) > 1:
            skipped_multiple_evidence += 1
            multiple_evidence_doc_names[doc_name] += 1
            continue

        questions_with_single_evidence_and_available_pdf += 1
        single_evidence_per_pdf[doc_name] += 1

    # Final retained query summary
    retained_questions = 0
    retained_questions_per_pdf = Counter()
    retained_question_type_counts = Counter()
    retained_question_type_per_pdf = {}

    skipped_from_query_file = 0
    skipped_reasons_from_query_file = Counter()

    for row in retained_query_rows:
        doc_name = row.get("doc_name")
        question_type = row.get("question_type", "UNKNOWN")

        kept_for_retrieval = row.get("kept_for_retrieval", True)
        ground_truth_chunk_id = row.get("ground_truth_chunk_id")

        is_retained = (
            kept_for_retrieval is True
            and ground_truth_chunk_id is not None
        )

        if not is_retained:
            skipped_from_query_file += 1
            skipped_reasons_from_query_file[row.get("skip_reason", "UNKNOWN")] += 1
            continue

        retained_questions += 1
        retained_questions_per_pdf[doc_name] += 1
        retained_question_type_counts[question_type] += 1

        if doc_name not in retained_question_type_per_pdf:
            retained_question_type_per_pdf[doc_name] = Counter()

        retained_question_type_per_pdf[doc_name][question_type] += 1

    # Questions that passed single-evidence filtering but were not retained
    # because evidence could not be matched to one usable chunk.
    unmatched_after_chunk_matching = (
        questions_with_single_evidence_and_available_pdf - retained_questions
    )

    retained_question_type_per_pdf_clean = {
        doc_name: dict(counts)
        for doc_name, counts in sorted(retained_question_type_per_pdf.items())
    }

    summary = {
        "pdf_folder": str(pdf_folder),
        "financebench_jsonl_file": str(financebench_jsonl),
        "queries_jsonl_file": str(queries_jsonl),

        "pdf_files_found": len(pdf_doc_names),

        "total_questions_in_original_jsonl": total_questions_in_jsonl,
        "questions_with_available_pdf": questions_with_available_pdf,
        "questions_with_single_evidence_and_available_pdf": questions_with_single_evidence_and_available_pdf,

        "questions_retained_after_chunk_matching": retained_questions,
        "questions_removed_after_chunk_matching": unmatched_after_chunk_matching,

        "questions_skipped_missing_pdf": skipped_missing_pdf,
        "questions_skipped_no_evidence": skipped_no_evidence,
        "questions_skipped_multiple_evidence": skipped_multiple_evidence,

        "retained_question_type_counts": dict(retained_question_type_counts),
        "retained_questions_per_pdf": dict(sorted(retained_questions_per_pdf.items())),
        "retained_question_type_per_pdf": retained_question_type_per_pdf_clean,

        "single_evidence_questions_per_pdf_before_chunk_matching": dict(sorted(single_evidence_per_pdf.items())),

        "missing_doc_names": dict(sorted(missing_doc_names.items())),
        "no_evidence_doc_names": dict(sorted(no_evidence_doc_names.items())),
        "multiple_evidence_doc_names": dict(sorted(multiple_evidence_doc_names.items())),

        "skipped_rows_inside_queries_file": skipped_from_query_file,
        "skipped_reasons_inside_queries_file": dict(skipped_reasons_from_query_file),
    }

    print("\n=== FINANCEBENCH QUESTION SUMMARY ===")
    print(f"PDF folder: {pdf_folder}")
    print(f"PDF files found: {len(pdf_doc_names)}")

    print("\n--- Original FinanceBench JSONL ---")
    print(f"Total questions in original JSONL: {total_questions_in_jsonl}")
    print(f"Questions with available PDF: {questions_with_available_pdf}")
    print(f"Questions with available PDF and exactly one evidence item: {questions_with_single_evidence_and_available_pdf}")
    print(f"Skipped because PDF is missing: {skipped_missing_pdf}")
    print(f"Skipped because evidence is missing: {skipped_no_evidence}")
    print(f"Skipped because evidence contains multiple items: {skipped_multiple_evidence}")

    print("\n--- Final retained query file ---")
    print(f"Retained questions after evidence-to-chunk matching: {retained_questions}")
    print(f"Removed after chunk matching: {unmatched_after_chunk_matching}")

    print("\nQuestion types among retained questions:")
    for question_type, count in sorted(retained_question_type_counts.items()):
        print(f"  {question_type}: {count}")

    print("\nRetained questions per PDF:")
    for doc_name, count in sorted(retained_questions_per_pdf.items()):
        print(f"  {doc_name}: {count}")

    print("\nQuestion types per PDF:")
    for doc_name, counts in sorted(retained_question_type_per_pdf.items()):
        counts_text = ", ".join(
            f"{question_type}: {count}"
            for question_type, count in sorted(counts.items())
        )
        print(f"  {doc_name}: {counts_text}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()