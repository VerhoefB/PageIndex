import argparse
import csv
from pathlib import Path

from pypdf import PdfReader


def count_pdf_pages(pdf_path):
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages), None
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Count the number of pages for each PDF in a folder."
    )

    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing PDF files"
    )

    parser.add_argument(
        "--output",
        default="pdf_page_counts.csv",
        help="Output CSV file"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders recursively"
    )

    args = parser.parse_args()

    folder = Path(args.folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if args.recursive:
        pdf_files = sorted(folder.rglob("*.pdf"))
    else:
        pdf_files = sorted(folder.glob("*.pdf"))

    rows = []

    total_pages = 0
    successful_pdfs = 0
    failed_pdfs = 0

    for pdf_path in pdf_files:
        page_count, error = count_pdf_pages(pdf_path)

        if page_count is not None:
            total_pages += page_count
            successful_pdfs += 1
        else:
            failed_pdfs += 1

        rows.append({
            "pdf_file": pdf_path.name,
            "pdf_path": str(pdf_path),
            "page_count": page_count if page_count is not None else "",
            "error": error if error else "",
        })

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pdf_file", "pdf_path", "page_count", "error"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== PDF PAGE COUNT SUMMARY ===")
    print(f"Folder: {folder}")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Successfully counted: {successful_pdfs}")
    print(f"Failed: {failed_pdfs}")
    print(f"Total pages: {total_pages}")

    if successful_pdfs > 0:
        print(f"Average pages per PDF: {total_pages / successful_pdfs:.2f}")

    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()