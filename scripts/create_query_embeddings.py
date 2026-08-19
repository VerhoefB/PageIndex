import argparse
import json
from pathlib import Path

import numpy as np
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
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding queries."
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embeddings for cosine similarity."
    )

    args = parser.parse_args()

    query_path = Path(args.queries)
    output_path = Path(args.output)

    rows = read_jsonl(query_path)
    queries = [get_query_text(row) for row in rows]

    print(f"Loaded {len(queries)} queries from {query_path}")
    print(f"Loading model: {args.model_name}")

    model = SentenceTransformer(args.model_name)

    print("Creating query embeddings...")

    embeddings = model.encode(
        queries,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    print(f"Saved query embeddings to: {output_path}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()