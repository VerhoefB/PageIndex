import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def get_query_key(row):
    financebench_id = row.get("financebench_id")

    # For FinanceBench, use the benchmark ID if it exists.
    if financebench_id not in (None, ""):
        return str(financebench_id)

    # For ESRS, use a stable content-based key for all methods.
    return (
        f"{row.get('doc_name', '')}__"
        f"{row.get('ground_truth_chunk_id', '')}__"
        f"{row.get('query', '')}"
    )

def normalize_chunk_ids(ids):
    if not ids:
        return []
    return [str(x) for x in ids]


def compute_metrics(row):
    gt = str(row.get("ground_truth_chunk_id", ""))

    top1_ids = normalize_chunk_ids(row.get("top1_chunk_ids"))
    top5_ids = normalize_chunk_ids(row.get("top5_chunk_ids"))

    correct_at_1 = int(len(top1_ids) > 0 and top1_ids[0] == gt)

    ground_truth_rank_top5 = None
    reciprocal_rank_at_5 = 0.0

    for rank, chunk_id in enumerate(top5_ids[:5], start=1):
        if chunk_id == gt:
            ground_truth_rank_top5 = rank
            reciprocal_rank_at_5 = 1.0 / rank
            break

    return correct_at_1, ground_truth_rank_top5, reciprocal_rank_at_5


def clean_method_name(row, fallback_name):
    method = row.get("method") or fallback_name
    model = row.get("model") or ""
    mode = row.get("mode") or ""

    return method, model, mode


def rows_to_metrics(path, dataset, fallback_name):
    raw_rows = load_jsonl(path)
    output_rows = []

    for row in raw_rows:
        if row.get("status") == "failed":
            continue
        if row.get("error"):
            continue

        query_key = get_query_key(row)

        if not query_key:
            continue

        if not row.get("ground_truth_chunk_id"):
            continue

        top1_ids = row.get("top1_chunk_ids") or []
        top5_ids = row.get("top5_chunk_ids") or []

        correct_at_1, rank_top5, rr5 = compute_metrics(row)
        method, model, mode = clean_method_name(row, fallback_name)

        output_rows.append({
            "dataset": dataset,
            "query_id": query_key,
            "company": row.get("company", ""),
            "doc_name": row.get("doc_name", ""),
            "bank_name": row.get("bank_name", ""),
            "method": method,
            "model": model,
            "mode": mode,
            "ground_truth_chunk_id": row.get("ground_truth_chunk_id"),
            "top1_chunk_id": top1_ids[0] if top1_ids else None,
            "top5_chunk_ids": "|".join(normalize_chunk_ids(top5_ids[:5])),
            "correct_at_1": correct_at_1,
            "ground_truth_rank_top5": rank_top5,
            "reciprocal_rank_at_5": rr5,
            "latency_seconds": row.get("latency_seconds"),
            "top1_latency_seconds": row.get("top1_latency_seconds"),
            "top5_latency_seconds": row.get("top5_latency_seconds"),
            "input_tokens": row.get("input_tokens", 0),
            "output_tokens": row.get("output_tokens", 0),
            "total_tokens": row.get("total_tokens", 0),
            "top1_total_tokens": row.get("top1_total_tokens", 0),
            "top5_total_tokens": row.get("top5_total_tokens", 0),
        })

    return pd.DataFrame(output_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output-detail", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--only-common-queries", action="store_true")
    args = parser.parse_args()

    if len(args.results) != len(args.names):
        raise ValueError("--results and --names must have the same length")

    dfs = []

    for path, name in zip(args.results, args.names):
        df = rows_to_metrics(path, args.dataset, name)
        df["result_file"] = name
        print(f"{name}: {len(df)} valid rows")
        dfs.append(df)

    detail = pd.concat(dfs, ignore_index=True)

    if args.only_common_queries:
        counts = detail.groupby("query_id")["result_file"].nunique()
        common_query_ids = counts[counts == len(args.results)].index
        detail = detail[detail["query_id"].isin(common_query_ids)].copy()
        print(f"Using only common queries: {len(common_query_ids)}")

    summary = (
        detail
        .groupby(["dataset", "result_file", "method", "model", "mode"], dropna=False)
        .agg(
            n_queries=("query_id", "nunique"),
            accuracy_at_1=("correct_at_1", "mean"),
            mrr_at_5=("reciprocal_rank_at_5", "mean"),
            avg_latency_seconds=("latency_seconds", "mean"),
            avg_top1_latency_seconds=("top1_latency_seconds", "mean"),
            avg_top5_latency_seconds=("top5_latency_seconds", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            avg_top1_total_tokens=("top1_total_tokens", "mean"),
            avg_top5_total_tokens=("top5_total_tokens", "mean"),
        )
        .reset_index()
    )

    Path(args.output_detail).parent.mkdir(parents=True, exist_ok=True)

    detail.to_csv(args.output_detail, index=False)
    summary.to_csv(args.output_summary, index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSaved detail to: {args.output_detail}")
    print(f"Saved summary to: {args.output_summary}")


if __name__ == "__main__":
    main()