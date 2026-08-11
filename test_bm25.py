import json

from baselines.bm25_retriever import BM25Retriever

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def evaluate_bm25(chunks_path, queries_path, top_k=5):
    chunks = load_jsonl(chunks_path)
    queries = load_jsonl(queries_path)

    retriever = BM25Retriever(chunks)

    total = 0
    correct_at_1 = 0
    reciprocal_rank_sum = 0.0

    for query_row in queries:
        query = query_row["query"]
        gold_chunk_id = str(query_row["ground_truth_chunk_id"])
        bank_name = query_row.get("bank_name")

        results = retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        retrieved_ids = [str(result["chunk_id"]) for result in results]

        total += 1

        if retrieved_ids and retrieved_ids[0] == gold_chunk_id:
            correct_at_1 += 1

        reciprocal_rank = 0.0

        for rank, retrieved_id in enumerate(retrieved_ids, start=1):
            if retrieved_id == gold_chunk_id:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_rank_sum += reciprocal_rank

    accuracy_at_1 = correct_at_1 / total if total else 0.0
    mrr_at_5 = reciprocal_rank_sum / total if total else 0.0

    print("=== BM25 EVALUATION ===")
    print(f"Queries: {total}")
    print(f"Accuracy@1: {accuracy_at_1:.4f}")
    print(f"MRR@{top_k}: {mrr_at_5:.4f}")


if __name__ == "__main__":
    evaluate_bm25(
        chunks_path=r".\results\ESRS chunks\Banque Postale_chunks.jsonl",
        queries_path=r".\results\ESRS queries\Banque Postale_queries.jsonl",
        top_k=5,
    )