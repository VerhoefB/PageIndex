import argparse
import json
import os
import re
from glob import glob
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


def unique_keep_order(values):
    seen = set()
    out = []

    for value in values:
        if value is None or value == "":
            continue

        if value not in seen:
            seen.add(value)
            out.append(value)

    return out


def doc_name_from_chunk_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    name = re.sub(r"_chunks$", "", name)
    return name


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
            if ch.get("doc_name") == row_doc_name
        ]

        if not candidate_chunks:
            report.append({
                "financebench_id": row.get("financebench_id"),
                "company": row.get("company"),
                "doc_name": row_doc_name,
                "query": row.get("question"),
                "answer": row.get("answer"),
                "evidence_index": None,
                "evidence_text": "",
                "matched_doc_name": None,
                "matched_chunk_id": None,
                "source_chunk_id": None,
                "matched_chunk_title": None,
                "match_type": "no_chunks_for_document",
                "score": 0.0,
                "line_coverage": 0.0,
                "number_coverage": 0.0,
                "lcs_ratio": 0.0,
                "same_doc": False,
                "kept": False,
            })
            continue

        candidate_chunks = [
            ch for ch in candidate_chunks
            if not ch.get("is_duplicate_chunk", False)
        ]

        evidence_matches = []

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

            same_doc = ch.get("doc_name") == row_doc_name

            kept = (
                same_doc
                and (
                    best["score"] >= min_score
                    or best["match_type"] == "evidence_contained_in_chunk"
                )
            )

            result = {
                "financebench_id": row.get("financebench_id"),
                "company": row.get("company"),
                "doc_name": row.get("doc_name"),
                "query": row.get("question"),
                "answer": row.get("answer"),
                "evidence_index": i,
                "evidence_text": ev_text,
                "matched_doc_name": ch.get("doc_name"),
                "matched_chunk_id": matched_chunk_id,
                "source_chunk_id": ch.get("chunk_id"),
                "matched_chunk_title": ch.get("title"),
                "match_type": best["match_type"],
                "score": best["score"],
                "line_coverage": best.get("line_coverage"),
                "number_coverage": best.get("number_coverage"),
                "lcs_ratio": best.get("lcs_ratio"),
                "same_doc": same_doc,
                "kept": kept,
            }

            report.append(result)

            if kept:
                evidence_matches.append({
                    "evidence_index": i,
                    "evidence_text": ev_text,
                    "matched_chunk_id": matched_chunk_id,
                    "source_chunk_id": ch.get("chunk_id"),
                    "matched_chunk_title": ch.get("title"),
                    "match_type": best["match_type"],
                    "match_score": best["score"],
                    "line_coverage": best.get("line_coverage"),
                    "number_coverage": best.get("number_coverage"),
                    "lcs_ratio": best.get("lcs_ratio"),
                })

        ground_truth_chunk_ids = unique_keep_order(
            m["matched_chunk_id"] for m in evidence_matches
        )

        matched_chunk_titles = unique_keep_order(
            m["matched_chunk_title"] for m in evidence_matches
        )

        keep_query = (
            len(row.get("evidence", [])) == 1
            and len(evidence_matches) == 1
            and len(ground_truth_chunk_ids) == 1
        )

        if keep_query:
            skip_reason = ""
        elif len(row.get("evidence", [])) != 1:
            skip_reason = "Question has multiple evidence entries."
        elif len(evidence_matches) == 0:
            skip_reason = "No evidence matched to a chunk."
        elif len(ground_truth_chunk_ids) != 1:
            skip_reason = "Evidence matched to zero or multiple chunks."
        else:
            skip_reason = "Not kept for retrieval."

        query_row = dict(row)
        query_row["query"] = row.get("question", "")
        query_row["ground_truth_chunk_ids"] = ground_truth_chunk_ids
        query_row["ground_truth_chunk_id"] = (
            ground_truth_chunk_ids[0] if ground_truth_chunk_ids else None
        )
        query_row["matched_chunk_titles"] = matched_chunk_titles
        query_row["evidence_matches"] = evidence_matches
        query_row["num_evidence"] = len(row.get("evidence", []))
        query_row["num_matched_evidence"] = len(evidence_matches)
        query_row["all_evidence_matched"] = (
            len(evidence_matches) == len(row.get("evidence", []))
            if row.get("evidence") else False
        )
        query_row["kept_for_retrieval"] = keep_query
        query_row["skip_reason"] = skip_reason

        if keep_query:
            output.append(query_row)

    write_jsonl(output, output_file)
    write_jsonl(report, report_file)

    print("FinanceBench rows checked:", len(fb_rows))
    print("Query rows written:", len(output))
    print("Report rows:", len(report))
    print("Saved:", output_file)
    print("Saved report:", report_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--financebench", required=True)
    p.add_argument("--chunks", required=True)
    p.add_argument("--output")
    p.add_argument("--report")
    p.add_argument("--doc_name")
    p.add_argument("--min_score", type=float, default=0.65)
    p.add_argument("--chunks_dir")
    p.add_argument("--output_dir")
    p.add_argument("--report_dir")
    p.add_argument("--combined_output")
    args = p.parse_args()

    if args.chunks_dir:
        if not args.report_dir:
            raise ValueError("--report_dir is required when using --chunks_dir")

        if not args.combined_output:
            raise ValueError("--combined_output is required when using --chunks_dir")

        chunk_files = sorted(glob(os.path.join(args.chunks_dir, "*_chunks.jsonl")))

        if not chunk_files:
            raise ValueError(f"No *_chunks.jsonl files found in {args.chunks_dir}")

        os.makedirs(args.report_dir, exist_ok=True)
        os.makedirs(os.path.dirname(args.combined_output) or ".", exist_ok=True)

        all_query_rows = []

        for chunk_file in chunk_files:
            doc_name = doc_name_from_chunk_filename(chunk_file)

            temp_output_file = os.path.join(
                os.path.dirname(args.combined_output) or ".",
                f"__tmp_{doc_name}_queries.jsonl"
            )

            report_file = os.path.join(
                args.report_dir,
                f"{doc_name}_evidence_match_report.jsonl"
            )

            print(f"\nMatching evidence for: {doc_name}")

            main(
                financebench_file=args.financebench,
                chunks_file=args.chunks,
                output_file=temp_output_file,
                report_file=report_file,
                doc_name=doc_name,
                min_score=args.min_score,
            )

            doc_query_rows = load_jsonl(temp_output_file)
            all_query_rows.extend(doc_query_rows)

            if os.path.exists(temp_output_file):
                os.remove(temp_output_file)

        write_jsonl(all_query_rows, args.combined_output)

        print("\nCombined query file saved:", args.combined_output)
        print("Total query rows:", len(all_query_rows))

    else:
        if not args.output or not args.report:
            raise ValueError("--output and --report are required when not using --chunks_dir")

        main(
            financebench_file=args.financebench,
            chunks_file=args.chunks,
            output_file=args.output,
            report_file=args.report,
            doc_name=args.doc_name,
            min_score=args.min_score,
        )