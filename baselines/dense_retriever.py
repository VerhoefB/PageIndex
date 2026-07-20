import faiss
import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    """
    Dense embedding-based retriever.

    This retriever:
    - embeds all canonical chunks
    - normalizes embeddings
    - indexes them with FAISS IndexFlatIP
    - embeds the query
    - retrieves top-k chunks using inner product similarity
    """

    def __init__(self, chunks, model_name, batch_size=8, device=None):
        # Use only canonical retrieval chunks if duplicate flags exist.
        self.chunks = [
            chunk for chunk in chunks
            if not chunk.get("is_duplicate_chunk", False)
        ]

        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

        self.index = None
        self.embeddings = None

    def _safe_model_name(self):
        return self.model_name.replace("/", "__").replace("-", "_")

    def _get_text(self, chunk):
        return chunk.get("text") or chunk.get("page_content") or ""

    def _get_chunk_id(self, chunk):
        return (
            chunk.get("retrieval_chunk_id")
            or chunk.get("canonical_chunk_id")
            or chunk.get("chunk_id")
        )

    def build_index(self, cache_path=None):
        """
        Encode all chunks and build FAISS index.

        If cache_path is provided and exists, embeddings are loaded from cache.
        Otherwise, embeddings are computed and saved to cache_path.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)

        if cache_path is not None and cache_path.exists():
            print(f"Loading embeddings from cache: {cache_path}")
            embeddings = np.load(cache_path)
        else:
            texts = [self._get_text(chunk) for chunk in self.chunks]

            print(f"Encoding {len(texts)} chunks...")

            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            embeddings = embeddings.astype("float32")

            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, embeddings)
                print(f"Saved embeddings to cache: {cache_path}")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype("float32"))

        self.embeddings = embeddings
        self.index = index

        print(f"FAISS index built with {index.ntotal} vectors.")
        print(f"Embedding dimension: {dimension}")

    def retrieve(self, query, top_k=5):
        """
        Retrieve top-k chunks for a query across the full corpus.
        """
        if self.index is None:
            raise ValueError("Index has not been built yet. Call build_index() first.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, idx in enumerate(indices[0], start=1):
            if idx == -1:
                continue

            chunk = self.chunks[idx]

            doc_name = chunk.get("doc_name") or chunk.get("bank_name") or ""

            results.append({
                "rank": rank,
                "chunk_id": str(self._get_chunk_id(chunk)),
                "source_chunk_id": str(chunk.get("chunk_id")),
                "doc_name": doc_name,
                "bank_name": chunk.get("bank_name", ""),
                "title": chunk.get("title", ""),
                "heading": chunk.get("heading", ""),
                "score": float(scores[0][rank - 1]),
                "text": self._get_text(chunk),
                "retriever": "dense",
            })

        return results