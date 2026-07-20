import argparse
import json
import os
import time

from baselines.bm25_retriever import BM25Retriever


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_ground_truth_rank(gold_chunk_id, retrieved_ids):
    gold_chunk_id = str(gold_chunk_id)

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if str(retrieved_id) == gold_chunk_id:
            return rank

    return None


def evaluate_bm25(
    chunks_path,
    queries_path,
    output_path,
    top_k=5,
    max_queries=None,
):
    chunks = load_jsonl(chunks_path)
    queries = load_jsonl(queries_path)

    if max_queries is not None:
        queries = queries[:max_queries]

    retriever = BM25Retriever(chunks)

    result_rows = []

    for query_row in queries:
        query = query_row["query"]
        gold_chunk_id = str(query_row["ground_truth_chunk_id"])

        start = time.time()

        results = retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        latency = time.time() - start

        top1_results = results[:1]
        top5_results = results[:5]

        top1_ids = [str(result["chunk_id"]) for result in top1_results]
        top5_ids = [str(result["chunk_id"]) for result in top5_results]

        ground_truth_rank_top1 = get_ground_truth_rank(gold_chunk_id, top1_ids)
        ground_truth_rank_top5 = get_ground_truth_rank(gold_chunk_id, top5_ids)

        correct_at_1 = 1 if ground_truth_rank_top1 == 1 else 0

        reciprocal_rank_at_5 = (
            1.0 / ground_truth_rank_top5
            if ground_truth_rank_top5 is not None
            else 0.0
        )

        result_rows.append({
            "financebench_id": query_row.get("financebench_id"),
            "company": query_row.get("company"),
            "doc_name": query_row.get("doc_name") or query_row.get("bank_name"),
            "bank_name": query_row.get("bank_name", ""),
            "query": query,
            "answer": query_row.get("answer"),
            "ground_truth_chunk_id": gold_chunk_id,

            "method": "bm25",
            "model": None,
            "mode": "standard",

            "top1_chunk_ids": top1_ids,
            "top5_chunk_ids": top5_ids,
            "top1_chunks": top1_results,
            "top5_chunks": top5_results,

            "correct_at_1": correct_at_1,
            "ground_truth_rank_top1": ground_truth_rank_top1,
            "ground_truth_rank_top5": ground_truth_rank_top5,
            "reciprocal_rank_at_5": reciprocal_rank_at_5,

            "top1_latency_seconds": round(latency, 3),
            "top5_latency_seconds": round(latency, 3),
            "latency_seconds": round(latency, 3),

            "top1_input_tokens": 0,
            "top1_output_tokens": 0,
            "top1_total_tokens": 0,

            "top5_input_tokens": 0,
            "top5_output_tokens": 0,
            "top5_total_tokens": 0,

            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        })

    write_jsonl(result_rows, output_path)

    print("=== BM25 QUERY-LEVEL EVALUATION SAVED ===")
    print(f"Queries: {len(result_rows)}")
    print(f"Output: {output_path}")

    if result_rows:
        accuracy_at_1 = sum(row["correct_at_1"] for row in result_rows) / len(result_rows)
        mrr_at_5 = sum(row["reciprocal_rank_at_5"] for row in result_rows) / len(result_rows)
        avg_latency = sum(row["latency_seconds"] for row in result_rows) / len(result_rows)

        print(f"Accuracy@1: {accuracy_at_1:.4f}")
        print(f"MRR@5: {mrr_at_5:.4f}")
        print(f"Average latency: {avg_latency:.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--chunks", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=None)

    args = parser.parse_args()

    evaluate_bm25(
        chunks_path=args.chunks,
        queries_path=args.queries,
        output_path=args.output,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )