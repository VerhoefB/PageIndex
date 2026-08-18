import argparse
import json
from collections import Counter
from pathlib import Path


def load_pdf_doc_names(pdf_folder: Path):
    return {
        pdf_path.stem
        for pdf_path in pdf_folder.glob("*.pdf")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Count FinanceBench questions matching available PDFs and single-evidence questions."
    )

    parser.add_argument(
        "--pdf-folder",
        required=True,
        help="Folder containing the FinanceBench PDF files, e.g. data/FinanceBench"
    )

    parser.add_argument(
        "--jsonl",
        required=True,
        help="FinanceBench open-source JSONL file"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON file with the summary"
    )

    args = parser.parse_args()

    pdf_folder = Path(args.pdf_folder)
    jsonl_path = Path(args.jsonl)

    if not pdf_folder.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder}")

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    pdf_doc_names = load_pdf_doc_names(pdf_folder)

    total_questions_in_jsonl = 0
    kept_questions = 0
    skipped_missing_pdf = 0
    skipped_no_evidence = 0
    skipped_multiple_evidence = 0

    question_type_counts = Counter()
    questions_per_pdf = Counter()
    question_type_per_pdf = {}

    missing_doc_names = Counter()
    no_evidence_doc_names = Counter()
    multiple_evidence_doc_names = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            total_questions_in_jsonl += 1
            row = json.loads(line)

            doc_name = row.get("doc_name")
            question_type = row.get("question_type", "UNKNOWN")
            evidence = row.get("evidence", [])

            if doc_name not in pdf_doc_names:
                skipped_missing_pdf += 1
                missing_doc_names[doc_name] += 1
                continue

            if not isinstance(evidence, list) or len(evidence) == 0:
                skipped_no_evidence += 1
                no_evidence_doc_names[doc_name] += 1
                continue

            if len(evidence) > 1:
                skipped_multiple_evidence += 1
                multiple_evidence_doc_names[doc_name] += 1
                continue

            kept_questions += 1
            question_type_counts[question_type] += 1
            questions_per_pdf[doc_name] += 1

            if doc_name not in question_type_per_pdf:
                question_type_per_pdf[doc_name] = Counter()

            question_type_per_pdf[doc_name][question_type] += 1

    total_skipped = skipped_missing_pdf + skipped_no_evidence + skipped_multiple_evidence

    question_type_per_pdf_clean = {
        doc_name: dict(counts)
        for doc_name, counts in sorted(question_type_per_pdf.items())
    }

    summary = {
        "pdf_folder": str(pdf_folder),
        "jsonl_file": str(jsonl_path),
        "pdf_files_found": len(pdf_doc_names),
        "total_questions_in_jsonl": total_questions_in_jsonl,
        "questions_kept": kept_questions,
        "questions_skipped_total": total_skipped,
        "questions_skipped_missing_pdf": skipped_missing_pdf,
        "questions_skipped_no_evidence": skipped_no_evidence,
        "questions_skipped_multiple_evidence": skipped_multiple_evidence,
        "question_type_counts_kept": dict(question_type_counts),
        "questions_per_pdf": dict(sorted(questions_per_pdf.items())),
        "question_type_per_pdf": question_type_per_pdf_clean,
        "missing_doc_names": dict(sorted(missing_doc_names.items())),
        "no_evidence_doc_names": dict(sorted(no_evidence_doc_names.items())),
        "multiple_evidence_doc_names": dict(sorted(multiple_evidence_doc_names.items())),
    }

    print("\n=== FINANCEBENCH QUESTION SUMMARY ===")
    print(f"PDF folder: {pdf_folder}")
    print(f"PDF files found: {len(pdf_doc_names)}")
    print(f"Total questions in JSONL: {total_questions_in_jsonl}")
    print(f"Questions kept: {kept_questions}")
    print(f"Questions skipped total: {total_skipped}")
    print(f"  Skipped because PDF is missing: {skipped_missing_pdf}")
    print(f"  Skipped because evidence is missing: {skipped_no_evidence}")
    print(f"  Skipped because evidence contains multiple items: {skipped_multiple_evidence}")

    print("\nQuestion types among kept questions:")
    for question_type, count in sorted(question_type_counts.items()):
        print(f"  {question_type}: {count}")

    print("\nQuestions per PDF:")
    for doc_name, count in sorted(questions_per_pdf.items()):
        print(f"  {doc_name}: {count}")

    print("\nQuestion types per PDF:")
    for doc_name, counts in sorted(question_type_per_pdf.items()):
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