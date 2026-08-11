import argparse
import json
import os
import time
import csv
from datetime import datetime

from baselines.dense_retriever import DenseRetriever


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

def append_csv_row(row, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    file_exists = os.path.exists(path)

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def get_ground_truth_rank(gold_chunk_id, retrieved_ids):
    gold_chunk_id = str(gold_chunk_id)

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if str(retrieved_id) == gold_chunk_id:
            return rank

    return None


def safe_model_label(model_name):
    return (
        model_name
        .replace("/", "__")
        .replace("-", "_")
        .replace(".", "_")
    )


def infer_dataset_from_path(path):
    normalized = path.replace("\\", "/").lower()

    if "esrs" in normalized:
        return "ESRS"

    if "financebench" in normalized:
        return "FinanceBench"

    return ""


def evaluate_dense(
    chunks_path,
    queries_path,
    model_name,
    output_path,
    cache_path=None,
    setup_csv_path=None,
    top_k=5,
    batch_size=8,
    max_queries=None,
):
    chunks = load_jsonl(chunks_path)
    queries = load_jsonl(queries_path)

    if max_queries is not None:
        queries = queries[:max_queries]

    # ------------------------------------------------------------
    # 1. Setup phase: load model, create/load embeddings, build index
    # ------------------------------------------------------------
    setup_start = time.time()
    status = "success"
    error = ""

    cache_loaded = bool(cache_path and os.path.exists(cache_path))

    try:
        retriever_start = time.time()

        retriever = DenseRetriever(
            chunks=chunks,
            model_name=model_name,
            batch_size=batch_size,
        )

        model_load_seconds = time.time() - retriever_start

        index_start = time.time()

        retriever.build_index(cache_path=cache_path)

        index_build_total_seconds = time.time() - index_start
        total_setup_seconds = time.time() - setup_start

    except Exception as e:
        status = "failed"
        error = str(e)
        total_setup_seconds = time.time() - setup_start

        if setup_csv_path is not None:
            append_csv_row({
                "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": infer_dataset_from_path(chunks_path),
                "method": "dense",
                "model": model_name,
                "chunks_path": chunks_path,
                "cache_path": cache_path or "",
                "num_chunks": len(chunks),
                "embedding_dimension": "",
                "batch_size": batch_size,
                "cache_loaded": cache_loaded,
                "model_load_seconds": round(model_load_seconds, 3) if "model_load_seconds" in locals() else "",
                "embedding_seconds": "",
                "cache_load_seconds": "",
                "index_build_seconds": "",
                "total_setup_seconds": round(total_setup_seconds, 3),
                "status": status,
                "error": error,
            }, setup_csv_path)

        raise

    # ------------------------------------------------------------
    # 2. Save one setup row: embedding/index construction tracking
    # ------------------------------------------------------------
    embedding_dimension = ""

    if hasattr(retriever, "embeddings"):
        try:
            embedding_dimension = retriever.embeddings.shape[1]
        except Exception:
            embedding_dimension = ""

    if cache_loaded:
        embedding_seconds = 0
        cache_load_seconds = index_build_total_seconds
    else:
        embedding_seconds = index_build_total_seconds
        cache_load_seconds = 0

    if setup_csv_path is not None:
        append_csv_row({
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": infer_dataset_from_path(chunks_path),
            "method": "dense",
            "model": model_name,
            "chunks_path": chunks_path,
            "cache_path": cache_path or "",
            "num_chunks": len(chunks),
            "embedding_dimension": embedding_dimension,
            "batch_size": batch_size,
            "cache_loaded": cache_loaded,
            "model_load_seconds": round(model_load_seconds, 3),
            "embedding_seconds": round(embedding_seconds, 3),
            "cache_load_seconds": round(cache_load_seconds, 3),
            "index_build_seconds": round(index_build_total_seconds, 3),
            "total_setup_seconds": round(total_setup_seconds, 3),
            "status": status,
            "error": error,
        }, setup_csv_path)

    # ------------------------------------------------------------
    # 3. Query-level retrieval evaluation
    # ------------------------------------------------------------
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

            "method": "dense",
            "model": model_name,
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

    print("\n=== DENSE QUERY-LEVEL EVALUATION SAVED ===")
    print(f"Model: {model_name}")
    print(f"Queries: {len(result_rows)}")
    print(f"Output: {output_path}")

    if setup_csv_path is not None:
        print(f"Setup CSV: {setup_csv_path}")

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
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--setup-csv", default=None)

    args = parser.parse_args()

    evaluate_dense(
        chunks_path=args.chunks,
        queries_path=args.queries,
        model_name=args.model_name,
        output_path=args.output,
        cache_path=args.cache_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_queries=args.max_queries,
        setup_csv_path=args.setup_csv
    )