import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer


class HybridPageIndexChunkRerankRetriever:
    """
    Hybrid PageIndex retrieval with chunk-text reranking.
    """

    def __init__(
        self,
        tree,
        chunks=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        device=None,
        node_cache_path=None,
        top_node_cache_path=None,
        chunk_cache_path=None,
    ):
        self.tree = tree
        self.chunks = chunks or []
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Loading hybrid PageIndex chunk-rerank model: {model_name}")
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

        self.nodes_by_id = {}
        self.node_embeddings = {}
        self.top_node_embeddings = {}

        self.chunk_lookup = self._build_chunk_lookup(self.chunks)
        self.chunk_embeddings = {}

        self._prepare_tree(
            node_cache_path=node_cache_path,
            top_node_cache_path=top_node_cache_path,
        )

        self._prepare_chunk_embeddings(
            chunk_cache_path=chunk_cache_path,
        )

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
        """Create text for node embeddings."""
        title = str(node.get("title", "") or "").strip()
        summary = str(node.get("summary", "") or "").strip()

        return f"{title}\n{summary}".strip()

    def _top_node_text(self, node):
        """Create text for document selection."""
        title = str(node.get("title", "") or "").strip()
        summary = str(node.get("summary", "") or "").strip()
        doc_description = str(node.get("doc_description", "") or "").strip()

        parts = [
            title,
            summary,
            doc_description,
        ]

        return "\n".join(part for part in parts if part)

    def _chunk_text(self, chunk):
        """Create chunk text for reranking."""
        title = str(chunk.get("title", "") or "").strip()
        text = str(chunk.get("text") or chunk.get("page_content") or "").strip()

        parts = [
            title,
            text,
        ]

        return "\n".join(part for part in parts if part)

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

    def _prepare_tree(self, node_cache_path=None, top_node_cache_path=None):
        node_ids = []
        node_texts = []

        top_node_ids = []
        top_node_texts = []

        roots = self.tree if isinstance(self.tree, list) else [self.tree]

        # Node embeddings
        for node_id, node in self._iter_nodes(self.tree):
            node["_hybrid_node_id"] = node_id
            self.nodes_by_id[node_id] = node

            node_ids.append(node_id)
            node_texts.append(self._node_text(node))

        document_roots = []

        for root in roots:
            if not isinstance(root, dict):
                continue

            if root.get("node_type") == "collection_root":
                document_roots.extend(self._get_children(root))
            else:
                document_roots.append(root)

        # Document embeddings
        for root_index, document_root in enumerate(document_roots):
            if not isinstance(document_root, dict):
                continue

            root_id = (
                document_root.get("_hybrid_node_id")
                or self._node_id(document_root, f"document_root_{root_index}")
            )

            document_root["_hybrid_node_id"] = str(root_id)

            top_node_ids.append(str(root_id))
            top_node_texts.append(self._top_node_text(document_root))

        if node_cache_path is not None:
            node_cache_path = Path(node_cache_path)

        if top_node_cache_path is not None:
            top_node_cache_path = Path(top_node_cache_path)

        # Load node embeddings
        if node_cache_path is not None and node_cache_path.exists():
            print(f"Loading hybrid node embeddings from cache: {node_cache_path}")
            embeddings = np.load(node_cache_path)
        else:
            print(f"Encoding {len(node_texts)} PageIndex nodes...")

            embeddings = self.model.encode(
                node_texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            if node_cache_path is not None:
                node_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(node_cache_path, embeddings)
                print(f"Saved hybrid node embeddings to cache: {node_cache_path}")

        if embeddings.shape[0] != len(node_ids):
            raise ValueError(
                f"Hybrid node embedding cache has {embeddings.shape[0]} rows, "
                f"but current tree has {len(node_ids)} nodes. "
                "Delete the cache file and rerun."
            )

        for node_id, embedding in zip(node_ids, embeddings):
            self.node_embeddings[node_id] = embedding

        # Load document embeddings
        if top_node_texts:
            if top_node_cache_path is not None and top_node_cache_path.exists():
                print(f"Loading hybrid top-node embeddings from cache: {top_node_cache_path}")
                top_embeddings = np.load(top_node_cache_path)
            else:
                print(f"Encoding {len(top_node_texts)} top-level document nodes...")

                top_embeddings = self.model.encode(
                    top_node_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype("float32")

                if top_node_cache_path is not None:
                    top_node_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(top_node_cache_path, top_embeddings)
                    print(f"Saved hybrid top-node embeddings to cache: {top_node_cache_path}")

            if top_embeddings.shape[0] != len(top_node_ids):
                raise ValueError(
                    f"Hybrid top-node embedding cache has {top_embeddings.shape[0]} rows, "
                    f"but current tree has {len(top_node_ids)} top nodes. "
                    "Delete the cache file and rerun."
                )

            for node_id, embedding in zip(top_node_ids, top_embeddings):
                self.top_node_embeddings[node_id] = embedding

        print("Hybrid PageIndex node embeddings ready.")

    def _prepare_chunk_embeddings(self, chunk_cache_path=None):
        chunk_ids = []
        chunk_texts = []

        for chunk in self.chunks:
            chunk_id = self._get_chunk_id(chunk)

            if chunk_id is None:
                continue

            chunk_ids.append(str(chunk_id))
            chunk_texts.append(self._chunk_text(chunk))

        if chunk_cache_path is not None:
            chunk_cache_path = Path(chunk_cache_path)

        if chunk_cache_path is not None and chunk_cache_path.exists():
            print(f"Loading hybrid chunk embeddings from cache: {chunk_cache_path}")
            chunk_embeddings = np.load(chunk_cache_path)
        else:
            print(f"Encoding {len(chunk_texts)} full chunk texts for reranking...")

            chunk_embeddings = self.model.encode(
                chunk_texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            if chunk_cache_path is not None:
                chunk_cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(chunk_cache_path, chunk_embeddings)
                print(f"Saved hybrid chunk embeddings to cache: {chunk_cache_path}")

        if chunk_embeddings.shape[0] != len(chunk_ids):
            raise ValueError(
                f"Hybrid chunk embedding cache has {chunk_embeddings.shape[0]} rows, "
                f"but current chunk file has {len(chunk_ids)} chunks. "
                "Delete the cache file and rerun."
            )

        for chunk_id, embedding in zip(chunk_ids, chunk_embeddings):
            self.chunk_embeddings[str(chunk_id)] = embedding

        print("Hybrid PageIndex chunk embeddings ready.")

    def _similarity(self, query_embedding, node):
        node_id = node.get("_hybrid_node_id")

        if node_id not in self.node_embeddings:
            return -1.0

        node_embedding = self.node_embeddings[node_id]

        return float(np.dot(query_embedding, node_embedding))

    def _top_similarity(self, query_embedding, node):
        node_id = node.get("_hybrid_node_id")

        if node_id not in self.top_node_embeddings:
            return -1.0

        node_embedding = self.top_node_embeddings[node_id]

        return float(np.dot(query_embedding, node_embedding))

    def _chunk_similarity(self, query_embedding, chunk_id):
        chunk_embedding = self.chunk_embeddings.get(str(chunk_id))

        if chunk_embedding is None:
            return None

        return float(np.dot(query_embedding, chunk_embedding))

    def _get_node_chunk_id(self, node):
        return (
            node.get("retrieval_chunk_id")
            or node.get("canonical_chunk_id")
            or node.get("chunk_id")
            or node.get("source_chunk_id")
        )

    def retrieve(self, query, top_k=5, top_m=2, query_embedding=None):
        """
        Traverse the tree using node embeddings and rerank the
        candidate chunks using full chunk-text embeddings.
        """
        if query_embedding is None:
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")[0]
        else:
            query_embedding = np.asarray(query_embedding, dtype="float32")

            if query_embedding.ndim == 2:
                query_embedding = query_embedding[0]

        roots = self.tree if isinstance(self.tree, list) else [self.tree]

       # Get document roots
        document_roots = []

        for root in roots:
            if not isinstance(root, dict):
                continue

            if root.get("node_type") == "collection_root":
                document_roots.extend(self._get_children(root))
            else:
                document_roots.append(root)

        # Select one document
        scored_document_roots = []

        for document_root in document_roots:
            score = self._top_similarity(query_embedding, document_root)
            scored_document_roots.append((score, document_root))

        scored_document_roots.sort(key=lambda x: x[0], reverse=True)

        frontier = [scored_document_roots[0][1]] if scored_document_roots else []
        final_candidates = []

        while frontier:
            pooled_children = []

            for node in frontier:
                children = self._get_children(node)

                # Store leaf candidates
                if not children:
                    chunk_id = self._get_node_chunk_id(node)

                    if chunk_id is not None:
                        final_candidates.append({
                            "chunk_id": str(chunk_id),
                            "node_id": node.get("_hybrid_node_id"),
                            "title": node.get("title", ""),
                            "summary": node.get("summary", ""),
                            "node_score": self._similarity(query_embedding, node),
                        })

                    continue

                for child in children:
                    score = self._similarity(query_embedding, child)
                    pooled_children.append((score, child))

            if not pooled_children:
                break

            # Keep the global top_m nodes
            pooled_children.sort(key=lambda x: x[0], reverse=True)
            selected_nodes = pooled_children[:top_m]

            next_frontier = []

            for score, child in selected_nodes:
                child_children = self._get_children(child)

                if not child_children:
                    chunk_id = self._get_node_chunk_id(child)

                    if chunk_id is not None:
                        final_candidates.append({
                            "chunk_id": str(chunk_id),
                            "node_id": child.get("_hybrid_node_id"),
                            "title": child.get("title", ""),
                            "summary": child.get("summary", ""),
                            "node_score": float(score),
                        })
                else:
                    next_frontier.append(child)

            frontier = next_frontier

        # Deduplicate candidates by chunk ID.
        best_by_chunk_id = {}

        for candidate in final_candidates:
            chunk_id = candidate["chunk_id"]

            if (
                chunk_id not in best_by_chunk_id
                or candidate["node_score"] > best_by_chunk_id[chunk_id]["node_score"]
            ):
                best_by_chunk_id[chunk_id] = candidate

        # Rerank using chunk embeddings
        reranked_candidates = []

        for chunk_id, candidate in best_by_chunk_id.items():
            chunk_score = self._chunk_similarity(query_embedding, chunk_id)

            if chunk_score is None:
                # Fallback to node score if chunk embedding is unavailable.
                final_score = float(candidate.get("node_score", 0.0))
                reranker = "node_score_fallback"
            else:
                final_score = float(chunk_score)
                reranker = "chunk_text_embedding"

            candidate = dict(candidate)
            candidate["score"] = final_score
            candidate["chunk_score"] = chunk_score
            candidate["reranker"] = reranker

            reranked_candidates.append(candidate)

        ranked = sorted(
            reranked_candidates,
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

        results = []

        for rank, item in enumerate(ranked, start=1):
            chunk_id = item["chunk_id"]
            chunk = self.chunk_lookup.get(str(chunk_id), {})

            doc_name = chunk.get("doc_name") or chunk.get("bank_name") or ""

            results.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "source_chunk_id": str(chunk.get("chunk_id", chunk_id)),
                "doc_name": doc_name,
                "bank_name": chunk.get("bank_name", ""),
                "node_id": item.get("node_id"),
                "title": item.get("title") or chunk.get("title", ""),
                "summary": item.get("summary", ""),
                "score": float(item["score"]),
                "node_score": float(item.get("node_score", 0.0)),
                "chunk_score": (
                    float(item["chunk_score"])
                    if item.get("chunk_score") is not None
                    else None
                ),
                "reranker": item.get("reranker"),
                "text": chunk.get("text") or chunk.get("page_content", ""),
                "retriever": "hybrid_pageindex_chunk_rerank",
            })

        return results