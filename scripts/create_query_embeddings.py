import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def get_query_text(row):
    for key in ["query", "question", "user_question"]:
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(f"No query field found in row: {row.keys()}")


def main():
    parser = argparse.ArgumentParser(
        description="Create cached query embeddings for dense retrieval."
    )

    parser.add_argument(
        "--queries",
        required=True,
        help="Path to query JSONL file."
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Embedding model name."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy file for query embeddings."
    )

    parser.add_argument(
        "--timing-output",
        required=True,
        help="Output CSV file for query embedding times."
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embeddings for cosine similarity."
    )

    args = parser.parse_args()

    query_path = Path(args.queries)
    output_path = Path(args.output)
    timing_output_path = Path(args.timing_output)

    rows = read_jsonl(query_path)
    queries = [get_query_text(row) for row in rows]

    print(f"Loaded {len(queries)} queries from {query_path}")
    print(f"Loading model: {args.model_name}")

    model = SentenceTransformer(args.model_name)

    print("Creating query embeddings and measuring query times...")

    embeddings = []
    timing_rows = []

    for i, (row, query) in enumerate(zip(rows, queries)):

        # Make sure previous GPU work is finished before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        embedding = model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=args.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

        # Make sure GPU computation has finished
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        embedding_time = time.perf_counter() - start_time

        embeddings.append(embedding[0])

        timing_rows.append({
            "query_index": i,
            "query": query,
            "financebench_id": row.get("financebench_id", ""),
            "doc_name": row.get("doc_name", row.get("bank_name", "")),
            "ground_truth_chunk_id": row.get("ground_truth_chunk_id", ""),
            "embedding_latency_seconds": embedding_time,
        })

        if (i + 1) % 25 == 0 or i == 0 or i + 1 == len(queries):
            print(
                f"[{i + 1}/{len(queries)}] "
                f"{embedding_time:.6f} seconds"
            )

    embeddings = np.vstack(embeddings).astype("float32")

    # Save embeddings exactly as before
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    # Save the additional timing information
    timing_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(timing_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_index",
                "query",
                "financebench_id",
                "doc_name",
                "ground_truth_chunk_id",
                "embedding_latency_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(timing_rows)

    print(f"Saved query embeddings to: {output_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Saved query embedding times to: {timing_output_path}")


if __name__ == "__main__":
    main()