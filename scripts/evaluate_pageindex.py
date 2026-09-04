import argparse
import json
import time
import os

from pageindex.pageindex_retriever import PageIndexLLMRetriever


GPT5_INPUT_PRICE_PER_1M = 1.25
GPT5_OUTPUT_PRICE_PER_1M = 10.00


def estimate_cost_usd(input_tokens, output_tokens):
    return (
        input_tokens / 1_000_000 * GPT5_INPUT_PRICE_PER_1M
        + output_tokens / 1_000_000 * GPT5_OUTPUT_PRICE_PER_1M
    )


def load_existing_results(output_path):
    completed_query_ids = set()

    total_input_tokens = 0
    total_output_tokens = 0
    completed_rows = 0

    if not os.path.exists(output_path):
        return completed_query_ids, completed_rows, total_input_tokens, total_output_tokens

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            query_id = row.get("query_id")

            if query_id is not None and not row.get("error"):
                completed_query_ids.add(str(query_id))
                completed_rows += 1
                total_input_tokens += int(row.get("input_tokens", 0) or 0)
                total_output_tokens += int(row.get("output_tokens", 0) or 0)

    return completed_query_ids, completed_rows, total_input_tokens, total_output_tokens


def write_jsonl_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_progress(
    progress_path,
    completed,
    total_queries,
    input_tokens,
    output_tokens,
    latest_query_id=None,
):
    cost_so_far = estimate_cost_usd(input_tokens, output_tokens)

    if completed > 0:
        avg_input = input_tokens / completed
        avg_output = output_tokens / completed

        projected_input = avg_input * total_queries
        projected_output = avg_output * total_queries
        projected_cost = estimate_cost_usd(projected_input, projected_output)
    else:
        avg_input = 0
        avg_output = 0
        projected_cost = 0

    progress = {
        "latest_query_id": latest_query_id,
        "completed_queries": completed,
        "total_queries": total_queries,
        "remaining_queries": max(total_queries - completed, 0),
        "input_tokens_so_far": input_tokens,
        "output_tokens_so_far": output_tokens,
        "cost_so_far_usd": round(cost_so_far, 6),
        "avg_input_tokens_per_query": round(avg_input, 2),
        "avg_output_tokens_per_query": round(avg_output, 2),
        "projected_total_cost_usd": round(projected_cost, 6),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(os.path.dirname(progress_path), exist_ok=True)

    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    return progress


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
    progress_path=None,
    max_cost_usd=None,
    stop_when_projected_cost_above=None,
):
    tree_json = load_json(tree_path)
    tree = get_tree_from_file(tree_json)

    chunks = load_jsonl(chunks_path)
    queries = load_jsonl(queries_path)

    if max_queries is not None:
        queries = queries[:max_queries]

    if progress_path is None:
        progress_path = output_path.replace(".jsonl", "_progress.json")

    completed_query_ids, completed_count, input_so_far, output_so_far = load_existing_results(output_path)

    print(f"Already completed: {completed_count}/{len(queries)}")
    print(f"Cost so far: ${estimate_cost_usd(input_so_far, output_so_far):.4f}")

    retriever = PageIndexLLMRetriever(
        tree=tree,
        chunks=chunks,
        model=model,
    )

    total = completed_count
    correct_at_1 = 0
    reciprocal_rank_sum = 0.0
    total_latency = 0.0
    top_k = 5 if mode in ["top5", "combined"] else 1

    for query_row in queries:
        query_id = str(
            query_row.get("query_id")
            or query_row.get("id")
            or query_row.get("ground_truth_chunk_id") + "_" + query_row.get("query", "")[:30]
        )

        if query_id in completed_query_ids:
            print(f"Skipping completed query: {query_id}")
            continue

        current_cost = estimate_cost_usd(input_so_far, output_so_far)

        if max_cost_usd is not None and current_cost >= max_cost_usd:
            print(f"Stopping safely: cost cap reached (${current_cost:.4f}).")
            break

        query = query_row["query"]
        gold_chunk_id = str(query_row["ground_truth_chunk_id"])

        try:
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

            top1_ids = [str(result["chunk_id"]) for result in top1_results]
            top5_ids = [str(result["chunk_id"]) for result in top5_results]

            ground_truth_rank_top1 = get_ground_truth_rank(gold_chunk_id, top1_ids)
            ground_truth_rank_top5 = get_ground_truth_rank(gold_chunk_id, top5_ids)

            query_correct_at_1 = 1 if ground_truth_rank_top1 == 1 else 0
            reciprocal_rank = 1.0 / ground_truth_rank_top5 if ground_truth_rank_top5 else 0.0

            result_row = {
                "query_id": query_id,

                "financebench_id": query_row.get("financebench_id"),
                "company": query_row.get("company"),
                "doc_name": query_row.get("doc_name"),
                "bank_name": query_row.get("bank_name", ""),
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

                "status": "completed",
            }

            input_so_far += int(result_row["input_tokens"] or 0)
            output_so_far += int(result_row["output_tokens"] or 0)
            total += 1
            correct_at_1 += query_correct_at_1
            reciprocal_rank_sum += reciprocal_rank
            total_latency += latency

            progress = write_progress(
                progress_path=progress_path,
                completed=total,
                total_queries=len(queries),
                input_tokens=input_so_far,
                output_tokens=output_so_far,
                latest_query_id=query_id,
            )

            result_row["cost_so_far_usd"] = progress["cost_so_far_usd"]
            result_row["projected_total_cost_usd"] = progress["projected_total_cost_usd"]

            write_jsonl_row(output_path, result_row)

            print("\nQUERY:", query)
            print("GOLD:", gold_chunk_id)
            print("TOP1:", top1_ids)
            print("TOP5:", top5_ids)
            print(f"LATENCY: {latency:.2f}s")
            print(f"COST SO FAR: ${progress['cost_so_far_usd']}")
            print(f"PROJECTED TOTAL COST: ${progress['projected_total_cost_usd']}")

            if (
                stop_when_projected_cost_above is not None
                and progress["projected_total_cost_usd"] > stop_when_projected_cost_above
            ):
                print(
                    f"Stopping safely: projected cost "
                    f"${progress['projected_total_cost_usd']} exceeds "
                    f"${stop_when_projected_cost_above}."
                )
                break

        except Exception as e:
            error_row = {
                "query_id": query_id,
                "query": query_row.get("query", ""),
                "ground_truth_chunk_id": query_row.get("ground_truth_chunk_id"),
                "method": "pageindex_llm",
                "model": model,
                "mode": mode,
                "status": "failed",
                "error": str(e),
            }

            write_jsonl_row(output_path, error_row)
            print(f"Failed query {query_id}: {e}")
            continue

    accuracy_at_1 = correct_at_1 / total if total else 0.0
    mrr = reciprocal_rank_sum / total if total else 0.0
    avg_latency = total_latency / total if total else 0.0

    print("\n=== PAGEINDEX LLM EVALUATION ===")
    print(f"Model: {model}")
    print(f"Mode: {mode}")
    print(f"Queries processed in this run/resume total: {total}")
    print(f"Accuracy@1 current run estimate: {accuracy_at_1:.4f}")
    print(f"MRR@{top_k} current run estimate: {mrr:.4f}")
    print(f"Average latency current run estimate: {avg_latency:.2f}s")
    print(f"Input tokens so far: {input_so_far}")
    print(f"Output tokens so far: {output_so_far}")
    print(f"Estimated cost so far: ${estimate_cost_usd(input_so_far, output_so_far):.4f}")
    print(f"Saved query-level results to: {output_path}")
    print(f"Saved progress to: {progress_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tree", required=True)
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--mode", choices=["top1", "top5", "combined"], default="combined")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-path", default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--stop-when-projected-cost-above", type=float, default=None)

    args = parser.parse_args()

    evaluate_pageindex_llm(
        tree_path=args.tree,
        chunks_path=args.chunks,
        queries_path=args.queries,
        model=args.model,
        mode=args.mode,
        output_path=args.output,
        max_queries=args.max_queries,
        progress_path=args.progress_path,
        max_cost_usd=args.max_cost_usd,
        stop_when_projected_cost_above=args.stop_when_projected_cost_above,
    )