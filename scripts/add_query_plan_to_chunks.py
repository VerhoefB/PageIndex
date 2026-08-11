import argparse
import json
import random
import re
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_text(row):
    return row.get("text") or row.get("page_content") or ""


def get_token_count(row):
    value = row.get("token_count") or row.get("text_token_count")
    if value is not None:
        try:
            return int(value)
        except Exception:
            pass

    # fallback estimate if token_count is missing
    return len(str(get_text(row)).split())


def normalize(value):
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def is_intro_chunk(node):
    chunk_id = str(node.get("chunk_id", "")).lower()
    node_id = str(node.get("node_id", "")).lower()
    title = str(node.get("title", "")).lower()

    return (
        bool(node.get("is_intro_node", False))
        or "intro" in chunk_id
        or "intro" in node_id
        or title.endswith("- introduction")
        or title.endswith(" introduction")
        or " introduction" in title
    )


def is_appendix_like(row):
    title = normalize(row.get("title", ""))
    heading = normalize(row.get("heading", ""))
    text_start = normalize(get_text(row)[:500])

    combined = f"{title} {heading} {text_start}"

    blocked_terms = [
        "appendix",
        "annex",
        "glossary",
        "assurance report",
        "limited assurance",
        "independent auditor",
        "table of contents",
        "contents",
        "index",
    ]

    return any(term in combined for term in blocked_terms)


def is_table_like(row):
    title = normalize(row.get("title", ""))
    heading = normalize(row.get("heading", ""))
    text = str(get_text(row) or "")

    if "table" in title.split() or "template" in title.split():
        return True

    if "table" in heading.split() or "template" in heading.split():
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    numeric_lines = 0
    percent_lines = 0
    wide_space_lines = 0

    for line in lines[:80]:
        if re.search(r"\d", line):
            numeric_lines += 1
        if "%" in line:
            percent_lines += 1
        if re.search(r"\s{3,}", line):
            wide_space_lines += 1

    n = min(len(lines), 80)

    numeric_ratio = numeric_lines / n
    percent_ratio = percent_lines / n
    wide_space_ratio = wide_space_lines / n

    return (
        numeric_ratio > 0.65
        and (wide_space_ratio > 0.25 or percent_ratio > 0.20)
    )


def is_duplicate(row):
    return bool(row.get("is_duplicate_chunk", False))


def is_eligible(row, min_tokens, min_chars):
    text = str(get_text(row) or "").strip()
    token_count = get_token_count(row)

    if is_duplicate(row):
        return False, "duplicate"

    if not text:
        return False, "empty"

    if len(text) < min_chars:
        return False, "too_short_chars"

    if token_count < min_tokens:
        return False, "too_short_tokens"

    if is_intro_chunk(row):
        return False, "intro"

    if is_appendix_like(row):
        return False, "appendix_like"

    if is_table_like(row):
        return False, "table_like"

    return True, ""


def assign_query_plan(rows, target_queries, seed, min_tokens, min_chars):
    random.seed(seed)

    eligible_indices = []

    for i, row in enumerate(rows):
        eligible, reason = is_eligible(row, min_tokens=min_tokens, min_chars=min_chars)

        row["generate_queries"] = False
        row["num_queries"] = 0
        row["skip_query_reason"] = reason

        if eligible:
            eligible_indices.append(i)

    if not eligible_indices:
        raise ValueError("No eligible chunks found for query generation.")

    random.shuffle(eligible_indices)

    base_queries = target_queries // len(eligible_indices)
    remainder = target_queries % len(eligible_indices)

    planned_queries = 0

    for position, row_index in enumerate(eligible_indices):
        n_queries = base_queries + (1 if position < remainder else 0)

        rows[row_index]["num_queries"] = n_queries
        rows[row_index]["generate_queries"] = n_queries > 0
        rows[row_index]["skip_query_reason"] = "" if n_queries > 0 else "not_selected"

        planned_queries += n_queries

    summary = {
        "total_chunks": len(rows),
        "eligible_chunks": len(eligible_indices),
        "target_queries": target_queries,
        "planned_queries": planned_queries,
        "base_queries_per_eligible_chunk": base_queries,
        "extra_query_chunks": remainder,
        "seed": seed,
        "min_tokens": min_tokens,
        "min_chars": min_chars,
    }

    return rows, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)

    parser.add_argument("--target-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tokens", type=int, default=100)
    parser.add_argument("--min-chars", type=int, default=300)

    args = parser.parse_args()

    rows = read_jsonl(args.chunks)

    rows, summary = assign_query_plan(
        rows=rows,
        target_queries=args.target_queries,
        seed=args.seed,
        min_tokens=args.min_tokens,
        min_chars=args.min_chars,
    )

    write_jsonl(args.output, rows)
    save_json(args.summary_output, summary)

    print("Query plan added.")
    print(f"Input chunks: {args.chunks}")
    print(f"Output chunks: {args.output}")
    print(f"Summary: {args.summary_output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()