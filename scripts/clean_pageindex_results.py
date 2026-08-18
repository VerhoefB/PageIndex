import argparse
import json
from pathlib import Path


def has_actual_result(row):
    """Keep only real result rows, remove failed/error-only rows."""

    # Failed PageIndex rows do not have the full result schema
    if "financebench_id" not in row:
        return False

    if row.get("status") == "failed":
        return False

    if row.get("error"):
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept_by_query_id = {}
    removed = 0
    total = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            total += 1
            row = json.loads(line)
            query_id = row.get("query_id")

            if not has_actual_result(row):
                removed += 1
                continue

            # If the same query appears multiple times, keep the latest valid row.
            kept_by_query_id[query_id] = row

    with output_path.open("w", encoding="utf-8") as f:
        for row in kept_by_query_id.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input rows: {total}")
    print(f"Removed failed/empty rows: {removed}")
    print(f"Kept valid rows: {len(kept_by_query_id)}")
    print(f"Saved cleaned file to: {output_path}")


if __name__ == "__main__":
    main()