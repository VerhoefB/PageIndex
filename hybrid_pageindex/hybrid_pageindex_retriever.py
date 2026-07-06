import numpy as np
import torch

from sentence_transformers import SentenceTransformer


class HybridPageIndexRetriever:
    """
    Hybrid PageIndex retriever.

    At each tree level:
    - embed query
    - compare query with child node title+summary embeddings
    - select top_m children
    - store selected leaf nodes
    - continue exploring selected non-leaf nodes
    """

    def __init__(
        self,
        tree,
        chunks=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        device=None,
    ):
        self.tree = tree
        self.chunks = chunks or []
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading hybrid PageIndex model: {model_name}")
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

        self.nodes_by_id = {}
        self.node_embeddings = {}
        self.chunk_lookup = self._build_chunk_lookup(self.chunks)

        self._prepare_tree()

    def _build_chunk_lookup(self, chunks):
        lookup = {}

        for chunk in chunks:
            chunk_id = self._get_chunk_id(chunk)

            if chunk_id is not None:
                lookup[str(chunk_id)] = chunk

        return lookup

    def _get_chunk_id(self, item):
        return (
            item.get("retrieval_chunk_id")
            or item.get("canonical_chunk_id")
            or item.get("chunk_id")
        )

    def _get_children(self, node):
        return node.get("nodes") or node.get("children") or []

    def _node_id(self, node, fallback):
        return str(node.get("node_id") or node.get("id") or fallback)

    def _node_text(self, node):
        """
        Only use title + summary for hybrid PageIndex node embeddings.
        """
        title = str(node.get("title", "") or "").strip()
        summary = str(node.get("summary", "") or "").strip()

        return f"{title}\n{summary}".strip()

    def _iter_nodes(self, nodes, prefix="root"):
        if isinstance(nodes, dict):
            nodes = [nodes]

        for i, node in enumerate(nodes or []):
            if not isinstance(node, dict):
                continue

            fallback_id = f"{prefix}_{i}"
            node_id = self._node_id(node, fallback_id)

            yield node_id, node

            for child_id, child in self._iter_nodes(
                self._get_children(node),
                prefix=node_id,
            ):
                yield child_id, child

    def _prepare_tree(self):
        node_ids = []
        node_texts = []

        for node_id, node in self._iter_nodes(self.tree):
            node["_hybrid_node_id"] = node_id
            self.nodes_by_id[node_id] = node

            node_ids.append(node_id)
            node_texts.append(self._node_text(node))

        print(f"Encoding {len(node_texts)} PageIndex nodes...")

        embeddings = self.model.encode(
            node_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        for node_id, embedding in zip(node_ids, embeddings):
            self.node_embeddings[node_id] = embedding

        print("Hybrid PageIndex node embeddings ready.")

    def _similarity(self, query_embedding, node):
        node_id = node.get("_hybrid_node_id")

        if node_id not in self.node_embeddings:
            return -1.0

        node_embedding = self.node_embeddings[node_id]

        return float(np.dot(query_embedding, node_embedding))

    def _get_node_chunk_id(self, node):
        return (
            node.get("retrieval_chunk_id")
            or node.get("canonical_chunk_id")
            or node.get("chunk_id")
            or node.get("source_chunk_id")
        )

    def retrieve(self, query, top_k=5, top_m=2):
        """
        Retrieve top-k chunks using hybrid PageIndex traversal.

        Algorithm:
        1. Start from the root nodes.
        2. For each explored node, score all children.
        3. Select top_m children.
        4. If a selected child is a leaf, store it.
        5. If a selected child has children, explore it next.
        6. Continue until no selected non-leaf nodes remain.
        7. Rank stored leaf chunks by their own title+summary similarity.
        """
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")[0]

        roots = self.tree if isinstance(self.tree, list) else [self.tree]

        frontier = list(roots)
        final_candidates = []

        while frontier:
            node = frontier.pop(0)
            children = self._get_children(node)

            # If current node is already a leaf, store it.
            if not children:
                chunk_id = self._get_node_chunk_id(node)

                if chunk_id is not None:
                    final_candidates.append({
                        "chunk_id": str(chunk_id),
                        "node_id": node.get("_hybrid_node_id"),
                        "title": node.get("title", ""),
                        "summary": node.get("summary", ""),
                        "score": self._similarity(query_embedding, node),
                    })

                continue

            scored_children = []

            for child in children:
                score = self._similarity(query_embedding, child)
                scored_children.append((score, child))

            scored_children.sort(key=lambda x: x[0], reverse=True)
            selected_children = scored_children[:top_m]

            for score, child in selected_children:
                child_children = self._get_children(child)

                if not child_children:
                    chunk_id = self._get_node_chunk_id(child)

                    if chunk_id is not None:
                        final_candidates.append({
                            "chunk_id": str(chunk_id),
                            "node_id": child.get("_hybrid_node_id"),
                            "title": child.get("title", ""),
                            "summary": child.get("summary", ""),
                            "score": float(score),
                        })
                else:
                    frontier.append(child)

        best_by_chunk_id = {}

        for candidate in final_candidates:
            chunk_id = candidate["chunk_id"]

            if (
                chunk_id not in best_by_chunk_id
                or candidate["score"] > best_by_chunk_id[chunk_id]["score"]
            ):
                best_by_chunk_id[chunk_id] = candidate

        ranked = sorted(
            best_by_chunk_id.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

        results = []

        for rank, item in enumerate(ranked, start=1):
            chunk_id = item["chunk_id"]
            chunk = self.chunk_lookup.get(str(chunk_id), {})

            results.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "source_chunk_id": str(chunk.get("chunk_id", chunk_id)),
                "bank_name": chunk.get("bank_name") or chunk.get("doc_name", ""),
                "node_id": item.get("node_id"),
                "title": item.get("title") or chunk.get("title", ""),
                "summary": item.get("summary", ""),
                "score": float(item["score"]),
                "text": chunk.get("text") or chunk.get("page_content", ""),
                "retriever": "hybrid_pageindex",
            })

        return results