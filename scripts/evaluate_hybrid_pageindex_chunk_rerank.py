import argparse
import csv
import json
import os
import time
from datetime import datetime

from hybrid_pageindex.hybrid_pageindex_chunk_rerank_retriever import HybridPageIndexChunkRerankRetriever


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def infer_dataset_from_path(path):
    normalized = path.replace("\\", "/").lower()

    if "esrs" in normalized:
        return "ESRS"

    if "financebench" in normalized:
        return "FinanceBench"

    return ""


def get_tree_from_file(tree_json):
    if isinstance(tree_json, dict) and "structure" in tree_json:
        return tree_json["structure"]

    return tree_json


def get_ground_truth_rank(gold_chunk_id, retrieved_ids):
    gold_chunk_id = str(gold_chunk_id)

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if str(retrieved_id) == gold_chunk_id:
            return rank

    return None


def evaluate_hybrid_pageindex(
    tree_path,
    chunks_path,
    queries_path,
    model_name,
    output_path,
    setup_csv_path=None,
    top_k=5,
    top_m=2,
    batch_size=8,
    max_queries=None,
    node_cache_path=None,
    top_node_cache_path=None,
    chunk_cache_path=None,
):
    tree_json = load_json(tree_path)
    tree = get_tree_from_file(tree_json)

    chunks = load_jsonl(chunks_path)
    queries = load_jsonl(queries_path)

    if max_queries is not None:
        queries = queries[:max_queries]

    # ------------------------------------------------------------
    # 1. Setup phase: load model, create/load node embeddings
    # ------------------------------------------------------------
    setup_start = time.time()
    status = "success"
    error = ""

    node_cache_loaded = bool(node_cache_path and os.path.exists(node_cache_path))
    top_node_cache_loaded = bool(top_node_cache_path and os.path.exists(top_node_cache_path))

    try:
        retriever_start = time.time()

        retriever = HybridPageIndexChunkRerankRetriever(
            tree=tree,
            chunks=chunks,
            model_name=model_name,
            batch_size=batch_size,
            node_cache_path=node_cache_path,
            top_node_cache_path=top_node_cache_path,
            chunk_cache_path=chunk_cache_path,
        )

        total_setup_seconds = time.time() - setup_start
        model_and_index_seconds = time.time() - retriever_start

    except Exception as e:
        status = "failed"
        error = str(e)
        total_setup_seconds = time.time() - setup_start

        if setup_csv_path is not None:
            append_csv_row({
                "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": infer_dataset_from_path(chunks_path),
                "method": "hybrid_pageindex",
                "model": model_name,
                "tree_path": tree_path,
                "chunks_path": chunks_path,
                "node_cache_path": node_cache_path or "",
                "top_node_cache_path": top_node_cache_path or "",
                "num_chunks": len(chunks),
                "num_nodes": "",
                "num_top_nodes": "",
                "embedding_dimension": "",
                "batch_size": batch_size,
                "top_k": top_k,
                "top_m": top_m,
                "node_cache_loaded": node_cache_loaded,
                "top_node_cache_loaded": top_node_cache_loaded,
                "node_embedding_seconds": "",
                "top_node_embedding_seconds": "",
                "cache_load_seconds": "",
                "model_and_index_seconds": "",
                "total_setup_seconds": round(total_setup_seconds, 3),
                "status": status,
                "error": error,
            }, setup_csv_path)

        raise

    # ------------------------------------------------------------
    # 2. Save one setup row: hybrid embedding/index tracking
    # ------------------------------------------------------------
    num_nodes = ""
    num_top_nodes = ""
    embedding_dimension = ""

    if hasattr(retriever, "node_embeddings"):
        try:
            num_nodes = len(retriever.node_embeddings)
        except Exception:
            num_nodes = ""

        try:
            first_embedding = next(iter(retriever.node_embeddings.values()))
            embedding_dimension = len(first_embedding)
        except Exception:
            embedding_dimension = ""

    if hasattr(retriever, "top_node_embeddings"):
        try:
            num_top_nodes = len(retriever.top_node_embeddings)
        except Exception:
            num_top_nodes = ""

    if node_cache_loaded and top_node_cache_loaded:
        node_embedding_seconds = 0
        top_node_embedding_seconds = 0
        cache_load_seconds = model_and_index_seconds
    else:
        node_embedding_seconds = model_and_index_seconds
        top_node_embedding_seconds = ""
        cache_load_seconds = 0

    if setup_csv_path is not None:
        append_csv_row({
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": infer_dataset_from_path(chunks_path),
            "method": "hybrid_pageindex",
            "model": model_name,
            "tree_path": tree_path,
            "chunks_path": chunks_path,
            "node_cache_path": node_cache_path or "",
            "top_node_cache_path": top_node_cache_path or "",
            "num_chunks": len(chunks),
            "num_nodes": num_nodes,
            "num_top_nodes": num_top_nodes,
            "embedding_dimension": embedding_dimension,
            "batch_size": batch_size,
            "top_k": top_k,
            "top_m": top_m,
            "node_cache_loaded": node_cache_loaded,
            "top_node_cache_loaded": top_node_cache_loaded,
            "node_embedding_seconds": round(node_embedding_seconds, 3) if node_embedding_seconds != "" else "",
            "top_node_embedding_seconds": round(top_node_embedding_seconds, 3) if top_node_embedding_seconds != "" else "",
            "cache_load_seconds": round(cache_load_seconds, 3),
            "model_and_index_seconds": round(model_and_index_seconds, 3),
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
            top_m=top_m,
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

            "method": "hybrid_pageindex",
            "model": model_name,
            "mode": "standard",
            "top_m": top_m,

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

        print("\nQUERY:", query)
        print("GOLD:", gold_chunk_id)
        print("TOP1:", top1_ids)
        print("TOP5:", top5_ids)
        print("CORRECT@1:", correct_at_1)
        print("RR@5:", reciprocal_rank_at_5)
        print(f"LATENCY: {latency:.3f}s")

    write_jsonl(result_rows, output_path)

    print("\n=== HYBRID PAGEINDEX QUERY-LEVEL EVALUATION SAVED ===")
    print(f"Model: {model_name}")
    print(f"Queries: {len(result_rows)}")
    print(f"top_m: {top_m}")
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

    parser.add_argument("--tree", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--node-cache-path", default=None)
    parser.add_argument("--top-node-cache-path", default=None)
    parser.add_argument("--setup-csv", default=None)
    parser.add_argument("--chunk-cache-path", default=None)

    args = parser.parse_args()

    evaluate_hybrid_pageindex(
        tree_path=args.tree,
        chunks_path=args.chunks,
        queries_path=args.queries,
        model_name=args.model_name,
        output_path=args.output,
        setup_csv_path=args.setup_csv,
        top_k=args.top_k,
        top_m=args.top_m,
        batch_size=args.batch_size,
        max_queries=args.max_queries,
        node_cache_path=args.node_cache_path,
        top_node_cache_path=args.top_node_cache_path,
        chunk_cache_path=args.chunk_cache_path,
    )