import argparse
import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher


def norm(t):
    t = t or ""
    t = t.lower().replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"[^a-z0-9]", "", t)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_evidence_text(ev):
    if isinstance(ev, str):
        return ev

    if isinstance(ev, dict):
        return ev.get("evidence_text") or ev.get("text") or ""

    return ""


def get_chunk_text(ch):
    return ch.get("text") or ch.get("page_content") or ""


def extract_numbers(text):
    """
    Extract financial/table-like numbers.

    Examples:
    - $55,893 -> 55893
    - (5,022) -> (5022)
    - -14.76% -> -14.76%
    - 0.62% -> 0.62%
    """
    text = text or ""

    raw = re.findall(
        r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?",
        text,
    )

    cleaned = []

    for x in raw:
        x = x.replace("$", "").replace(",", "").strip()

        if x:
            cleaned.append(x)

    return cleaned


def multiset_coverage(needles, haystack):
    """
    Coverage of evidence numbers by chunk numbers, respecting repeated values.
    """
    if not needles:
        return 0.0

    need = Counter(needles)
    have = Counter(haystack)

    matched = 0
    total = sum(need.values())

    for k, v in need.items():
        matched += min(v, have.get(k, 0))

    return matched / total if total else 0.0


def evidence_lines(evidence_text, min_norm_len=5):
    """
    Convert evidence into meaningful normalized lines.

    This is much better for financial-table evidence than comparing the entire
    evidence string with the entire chunk string.
    """
    lines = []

    for raw_line in (evidence_text or "").splitlines():
        raw_line = raw_line.strip()

        if not raw_line:
            continue

        n = norm(raw_line)

        if len(n) < min_norm_len:
            continue

        lines.append({
            "raw": raw_line,
            "norm": n,
            "weight": max(len(n), 1),
        })

    return lines


def weighted_line_coverage(evidence_text, chunk_text):
    """
    Weighted share of evidence lines found inside the chunk.

    Long label lines matter more than tiny lines like '2022'.
    """
    lines = evidence_lines(evidence_text)

    if not lines:
        return 0.0

    chunk_norm = norm(chunk_text)

    total_weight = sum(line["weight"] for line in lines)
    hit_weight = 0

    hits = []

    for line in lines:
        if line["norm"] in chunk_norm:
            hit_weight += line["weight"]
            hits.append(line["raw"])

    return hit_weight / total_weight if total_weight else 0.0


def longest_common_substring_ratio(evidence_text, chunk_text):
    """
    Finds the longest contiguous normalized overlap relative to the evidence.

    This helps when line breaks differ between FinanceBench evidence and chunks.
    """
    e = norm(evidence_text)
    c = norm(chunk_text)

    if not e or not c:
        return 0.0

    if e in c:
        return 1.0

    matcher = SequenceMatcher(None, e, c, autojunk=False)
    match = matcher.find_longest_match(0, len(e), 0, len(c))

    return match.size / len(e) if e else 0.0


def chunk_match_score(evidence_text, chunk):
    chunk_text = get_chunk_text(chunk)

    e_norm = norm(evidence_text)
    c_norm = norm(chunk_text)

    if not e_norm or not c_norm:
        return {
            "score": 0.0,
            "match_type": "empty",
            "line_coverage": 0.0,
            "number_coverage": 0.0,
            "lcs_ratio": 0.0,
        }

    if e_norm in c_norm:
        return {
            "score": 1.0,
            "match_type": "evidence_contained_in_chunk",
            "line_coverage": 1.0,
            "number_coverage": 1.0,
            "lcs_ratio": 1.0,
        }

    if c_norm in e_norm:
        contained_score = len(c_norm) / len(e_norm)

        return {
            "score": contained_score,
            "match_type": "chunk_contained_in_evidence",
            "line_coverage": 1.0,
            "number_coverage": 1.0,
            "lcs_ratio": contained_score,
        }

    line_cov = weighted_line_coverage(evidence_text, chunk_text)

    evidence_numbers = extract_numbers(evidence_text)
    chunk_numbers = extract_numbers(chunk_text)
    number_cov = multiset_coverage(evidence_numbers, chunk_numbers)

    lcs_ratio = longest_common_substring_ratio(evidence_text, chunk_text)

    # Main score:
    # - line coverage is most important for tables
    # - number coverage helps financial statements
    # - LCS helps when line breaks differ
    score = max(
        0.75 * line_cov + 0.25 * number_cov,
        0.65 * lcs_ratio + 0.35 * number_cov,
    )

    return {
        "score": score,
        "match_type": "partial_line_number_match",
        "line_coverage": line_cov,
        "number_coverage": number_cov,
        "lcs_ratio": lcs_ratio,
    }


def best_chunk_match(evidence_text, chunks):
    best = None

    for ch in chunks:
        stats = chunk_match_score(evidence_text, ch)

        if best is None or stats["score"] > best["score"]:
            best = {
                "chunk": ch,
                **stats,
            }

    return best


def main(
    financebench_file,
    chunks_file,
    output_file,
    report_file,
    doc_name=None,
    min_score=0.65,
):
    fb_rows = load_jsonl(financebench_file)
    chunks = load_jsonl(chunks_file)

    if doc_name:
        fb_rows = [r for r in fb_rows if r.get("doc_name") == doc_name]

    output, report = [], []

    for row in fb_rows:
        row_doc_name = row.get("doc_name")

        candidate_chunks = [
            ch for ch in chunks
            if not row_doc_name or ch.get("doc_name") == row_doc_name
        ]

        if not candidate_chunks:
            candidate_chunks = chunks

        # Do not match to duplicate chunks.
        candidate_chunks = [
            ch for ch in candidate_chunks
            if not ch.get("is_duplicate_chunk", False)
        ]

        for i, ev in enumerate(row.get("evidence", [])):
            ev_text = get_evidence_text(ev)
            best = best_chunk_match(ev_text, candidate_chunks)

            if not best:
                continue

            ch = best["chunk"]

            matched_chunk_id = (
                ch.get("retrieval_chunk_id")
                or ch.get("canonical_chunk_id")
                or ch.get("chunk_id")
            )

            kept = (
                best["score"] >= min_score
                or best["match_type"] == "evidence_contained_in_chunk"
            )

            result = {
                "financebench_id": row.get("financebench_id"),
                "company": row.get("company"),
                "doc_name": row.get("doc_name"),
                "query": row.get("question"),
                "answer": row.get("answer"),
                "evidence_index": i,
                "evidence_text": ev_text,
                "matched_chunk_id": matched_chunk_id,
                "source_chunk_id": ch.get("chunk_id"),
                "matched_chunk_title": ch.get("title"),
                "match_type": best["match_type"],
                "score": best["score"],
                "line_coverage": best.get("line_coverage"),
                "number_coverage": best.get("number_coverage"),
                "lcs_ratio": best.get("lcs_ratio"),
                "kept": kept,
            }

            report.append(result)

            if kept:
                merged = dict(row)
                merged["query"] = row.get("question", "")
                merged["ground_truth_chunk_id"] = matched_chunk_id
                merged["matched_chunk_id"] = matched_chunk_id
                merged["source_chunk_id"] = ch.get("chunk_id")
                merged["matched_chunk_title"] = ch.get("title")
                merged["matched_chunk_text"] = get_chunk_text(ch)
                merged["evidence_index"] = i
                merged["evidence_text"] = ev_text
                merged["match_type"] = best["match_type"]
                merged["match_score"] = best["score"]
                merged["line_coverage"] = best.get("line_coverage")
                merged["number_coverage"] = best.get("number_coverage")
                merged["lcs_ratio"] = best.get("lcs_ratio")
                output.append(merged)

    write_jsonl(output, output_file)
    write_jsonl(report, report_file)

    print("FinanceBench rows checked:", len(fb_rows))
    print("Matched evidence rows:", len(output))
    print("Report rows:", len(report))
    print("Saved:", output_file)
    print("Saved report:", report_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--financebench", required=True)
    p.add_argument("--chunks", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--doc_name")
    p.add_argument("--min_score", type=float, default=0.65)
    args = p.parse_args()

    main(
        args.financebench,
        args.chunks,
        args.output,
        args.report,
        args.doc_name,
        args.min_score,
    )