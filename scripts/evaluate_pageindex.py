import argparse
import json
import time
import csv
import os
from datetime import datetime

from pageindex.pageindex_retriever import PageIndexLLMRetriever


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


def get_ground_truth_rank(gold_chunk_id, retrieved_ids):
    gold_chunk_id = str(gold_chunk_id)

    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if str(retrieved_id) == gold_chunk_id:
            return rank

    return None


def usage_delta(before, after):
    return {
        "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        "total_tokens": after["total_tokens"] - before["total_tokens"],
    }

def get_tree_from_file(tree_json):
    if isinstance(tree_json, dict) and "structure" in tree_json:
        return tree_json["structure"]

    return tree_json

def evaluate_pageindex_llm(
    tree_path,
    chunks_path,
    queries_path,
    model,
    mode,
    output_path,
    max_queries=None,
):
    tree_json = load_json(tree_path)
    tree = get_tree_from_file(tree_json)

    chunks = load_jsonl(chunks_path)
    queries = load_jsonl(queries_path)

    if max_queries is not None:
        queries = queries[:max_queries]

    retriever = PageIndexLLMRetriever(
        tree=tree,
        chunks=chunks,
        model=model,
    )

    total = 0
    correct_at_1 = 0
    reciprocal_rank_sum = 0.0
    total_latency = 0.0

    result_rows = []

    for query_row in queries:
        query = query_row["query"]
        gold_chunk_id = str(query_row["ground_truth_chunk_id"])

        usage_before = retriever.get_usage()
        start = time.time()

        if mode == "top1":
            results = retriever.retrieve_top1(
                query=query,
                safety_max_chunk_reads=10,
            )

            top1_results = results
            top5_results = results
            top_k = 1

            usage_top1 = None
            usage_top5 = None
            usage_total = None


            latency_top1 = None
            latency_top5 = None
            latency_total_from_retriever = None

        elif mode == "top5":
            results = retriever.retrieve_top5(
                query=query,
                max_chunk_reads=5,
            )

            top1_results = results[:1]
            top5_results = results
            top_k = 5

            usage_top1 = None
            usage_top5 = None
            usage_total = None

            latency_top1 = None
            latency_top5 = None
            latency_total_from_retriever = None

        elif mode == "combined":
            combined = retriever.retrieve_combined(
                query=query,
                max_top5_reads=5,
                safety_max_chunk_reads=10,
            )

            top1_results = combined["top1"]
            top5_results = combined["top5"]
            top_k = 5

            usage_top1 = combined.get("usage_top1", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            })
            usage_top5 = combined.get("usage_top5", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            })
            usage_total = combined.get("usage_total", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            })
            latency_top1 = combined.get("latency_top1_seconds")
            latency_top5 = combined.get("latency_top5_seconds")
            latency_total_from_retriever = combined.get("latency_total_seconds")

        else:
            raise ValueError("mode must be 'top1', 'top5', or 'combined'")

        latency = time.time() - start
        usage_after = retriever.get_usage()

        if usage_total is None:
            usage_total = usage_delta(usage_before, usage_after)

        if usage_top1 is None:
            usage_top1 = usage_total if mode == "top1" else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        if usage_top5 is None:
            usage_top5 = usage_total if mode == "top5" else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        total_latency += latency

        top1_ids = [str(result["chunk_id"]) for result in top1_results]
        top5_ids = [str(result["chunk_id"]) for result in top5_results]

        total += 1

        if top1_ids and top1_ids[0] == gold_chunk_id:
            correct_at_1 += 1

        reciprocal_rank = 0.0

        for rank, retrieved_id in enumerate(top5_ids, start=1):
            if retrieved_id == gold_chunk_id:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_rank_sum += reciprocal_rank

        ground_truth_rank_top1 = get_ground_truth_rank(gold_chunk_id, top1_ids)
        ground_truth_rank_top5 = get_ground_truth_rank(gold_chunk_id, top5_ids)

        query_correct_at_1 = 1 if ground_truth_rank_top1 == 1 else 0

        result_rows.append({
            "financebench_id": query_row.get("financebench_id"),
            "company": query_row.get("company"),
            "doc_name": query_row.get("doc_name"),
            "query": query,
            "answer": query_row.get("answer"),
            "ground_truth_chunk_id": gold_chunk_id,

            "method": "pageindex_llm",
            "model": model,
            "mode": mode,

            "top1_chunk_ids": top1_ids,
            "top5_chunk_ids": top5_ids,
            "top1_chunks": top1_results,
            "top5_chunks": top5_results,

            "correct_at_1": query_correct_at_1,
            "ground_truth_rank_top1": ground_truth_rank_top1,
            "ground_truth_rank_top5": ground_truth_rank_top5,
            "reciprocal_rank_at_5": reciprocal_rank,

            "top1_latency_seconds": latency_top1 if latency_top1 is not None else (
                round(latency, 3) if mode == "top1" else None
            ),
            "top5_latency_seconds": latency_top5 if latency_top5 is not None else (
                round(latency, 3) if mode == "top5" else None
            ),
            "latency_seconds": (
                latency_total_from_retriever
                if latency_total_from_retriever is not None
                else round(latency, 3)
            ),

            "top1_input_tokens": usage_top1["prompt_tokens"],
            "top1_output_tokens": usage_top1["completion_tokens"],
            "top1_total_tokens": usage_top1["total_tokens"],

            "top5_input_tokens": usage_top5["prompt_tokens"],
            "top5_output_tokens": usage_top5["completion_tokens"],
            "top5_total_tokens": usage_top5["total_tokens"],

            "input_tokens": usage_total["prompt_tokens"],
            "output_tokens": usage_total["completion_tokens"],
            "total_tokens": usage_total["total_tokens"],
        })

        print("\nQUERY:", query)
        print("GOLD:", gold_chunk_id)
        print("TOP1:", top1_ids)
        print("TOP5:", top5_ids)
        print(f"LATENCY: {latency:.2f}s")

    accuracy_at_1 = correct_at_1 / total if total else 0.0
    mrr = reciprocal_rank_sum / total if total else 0.0
    avg_latency = total_latency / total if total else 0.0

    write_jsonl(result_rows, output_path)
    print(f"Saved query-level results to: {output_path}")
    usage = retriever.get_usage()

    print("\n=== PAGEINDEX LLM EVALUATION ===")
    print(f"Model: {model}")
    print(f"Mode: {mode}")
    print(f"Queries: {total}")
    print(f"Accuracy@1: {accuracy_at_1:.4f}")
    print(f"MRR@{top_k}: {mrr:.4f}")
    print(f"Average latency: {avg_latency:.2f}s")
    print(f"Prompt tokens: {usage['prompt_tokens']}")
    print(f"Completion tokens: {usage['completion_tokens']}")
    print(f"Total tokens: {usage['total_tokens']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tree", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--mode", choices=["top1", "top5", "combined"], default="combined")
    parser.add_argument("--max-queries", type=int, default=2)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    evaluate_pageindex_llm(
        tree_path=args.tree,
        chunks_path=args.chunks,
        queries_path=args.queries,
        model=args.model,
        mode=args.mode,
        output_path=args.output,
        max_queries=args.max_queries,
    )