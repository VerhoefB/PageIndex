import argparse, json, os, re
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

def best_chunk_match(evidence_text, chunks):
    e = norm(evidence_text)
    best = None

    for ch in chunks:
        c = norm(ch.get("text", ""))

        if not e or not c:
            continue

        if e in c:
            score, match_type = 1.0, "evidence_contained_in_chunk"
        elif c in e:
            score, match_type = len(c) / len(e), "chunk_contained_in_evidence"
        else:
            score = SequenceMatcher(None, e, c).ratio()
            match_type = "partial"

        if best is None or score > best["score"]:
            best = {
                "chunk": ch,
                "score": score,
                "match_type": match_type,
            }

    return best

def main(financebench_file, chunks_file, output_file, report_file, doc_name=None, min_score=0.8):
    fb_rows = load_jsonl(financebench_file)
    chunks = load_jsonl(chunks_file)

    if doc_name:
        fb_rows = [r for r in fb_rows if r.get("doc_name") == doc_name]

    output, report = [], []

    for row in fb_rows:
        for i, ev in enumerate(row.get("evidence", [])):
            ev_text = get_evidence_text(ev)
            best = best_chunk_match(ev_text, chunks)

            if not best:
                continue

            ch = best["chunk"]
            kept = best["score"] >= min_score or best["match_type"] == "evidence_contained_in_chunk"

            result = {
                "financebench_id": row.get("financebench_id"),
                "company": row.get("company"),
                "doc_name": row.get("doc_name"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "evidence_index": i,
                "evidence_text": ev_text,
                "matched_chunk_id": ch.get("chunk_id"),
                "matched_chunk_title": ch.get("title"),
                "match_type": best["match_type"],
                "score": best["score"],
                "kept": kept,
            }

            report.append(result)

            if kept:
                merged = dict(row)
                merged["matched_chunk_id"] = ch.get("chunk_id")
                merged["matched_chunk_title"] = ch.get("title")
                merged["matched_chunk_text"] = ch.get("text")
                merged["evidence_index"] = i
                merged["evidence_text"] = ev_text
                merged["match_type"] = best["match_type"]
                merged["match_score"] = best["score"]
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
    p.add_argument("--min_score", type=float, default=0.8)
    args = p.parse_args()

    main(args.financebench, args.chunks, args.output, args.report, args.doc_name, args.min_score)