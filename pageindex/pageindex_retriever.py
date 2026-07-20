import json
import os
import time
from tracemalloc import start
import tiktoken
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)


class PageIndexLLMRetriever:
    """
    LLM-based PageIndex retriever adapted for chunk-level evaluation.

    Recommended mode: retrieve_combined()
    - Uses one continuous traversal state.
    - Does not restart from root for every retrieved chunk.
    - Collects up to 5 unique chunks for MRR@5.
    - Records the first sufficient chunk as top1.
    - Does not stop before collecting the top-5 candidate set.
    """

    def __init__(
        self,
        tree: Any,
        chunks: List[Dict[str, Any]],
        model: str = "gpt-5",
        max_retries: int = 3,
        sleep_between_calls: float = 0.2,
        debug_token_counts: bool = True,
    ):
        self.tree = tree
        self.chunks = chunks
        self.model = model
        self.max_retries = max_retries
        self.sleep_between_calls = sleep_between_calls
        self.debug_token_counts = debug_token_counts

        self.nodes_by_id: Dict[str, Dict[str, Any]] = {}
        self.parent_by_id: Dict[str, Optional[str]] = {}
        self.chunk_lookup = self._build_chunk_lookup(chunks)

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

        self.encoding = tiktoken.get_encoding("o200k_base")

        self._index_tree()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    def _get_children(self, node: Dict[str, Any]) -> List[Dict[str, Any]]:
        return (
            node.get("nodes")
            or node.get("children")
            or node.get("sub_nodes")
            or []
        )

    def _get_chunk_id_from_item(self, item: Dict[str, Any]) -> Optional[str]:
        chunk_id = (
            item.get("retrieval_chunk_id")
            or item.get("canonical_chunk_id")
            or item.get("chunk_id")
            or item.get("source_chunk_id")
        )
        return str(chunk_id) if chunk_id is not None else None

    def _node_id(self, node: Dict[str, Any], fallback: str) -> str:
        return str(node.get("node_id") or node.get("id") or fallback)

    def _get_node_id_from_node(self, node: Dict[str, Any]) -> str:
        return str(
            node.get("_pageindex_node_id")
            or node.get("node_id")
            or node.get("id")
            or ""
        )

    def _get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.nodes_by_id.get(str(node_id))

    def _get_parent_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        parent_id = self.parent_by_id.get(str(node_id))
        if parent_id is None:
            return None
        return self.nodes_by_id.get(str(parent_id))

    def _is_leaf(self, node: Dict[str, Any]) -> bool:
        return len(self._get_children(node)) == 0

    def _build_chunk_lookup(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        lookup = {}
        for chunk in chunks:
            chunk_id = self._get_chunk_id_from_item(chunk)
            if chunk_id is not None:
                lookup[str(chunk_id)] = chunk
        return lookup

    def _iter_nodes(self, nodes: Any, parent_id: Optional[str] = None, prefix: str = "root"):
        if isinstance(nodes, dict):
            nodes = [nodes]

        for i, node in enumerate(nodes or []):
            if not isinstance(node, dict):
                continue

            fallback_id = f"{prefix}_{i}"
            node_id = self._node_id(node, fallback_id)

            yield node_id, node, parent_id

            for child_id, child, _ in self._iter_nodes(
                self._get_children(node),
                parent_id=node_id,
                prefix=node_id,
            ):
                yield child_id, child, node_id

    def _index_tree(self):
        roots = self.tree if isinstance(self.tree, list) else [self.tree]

        for node_id, node, parent_id in self._iter_nodes(roots):
            node["_pageindex_node_id"] = node_id
            self.nodes_by_id[node_id] = node
            self.parent_by_id[node_id] = parent_id

        print(f"Indexed PageIndex nodes: {len(self.nodes_by_id)}")
        print(f"Indexed chunks: {len(self.chunk_lookup)}")

    # ------------------------------------------------------------------
    # PageIndex-style tools / retrieval representations
    # ------------------------------------------------------------------

    def get_document(self) -> Dict[str, Any]:
        roots = self.tree if isinstance(self.tree, list) else [self.tree]
        return {
            "num_root_nodes": len(roots),
            "num_indexed_nodes": len(self.nodes_by_id),
            "num_chunks": len(self.chunk_lookup),
            "root_nodes": [
                {
                    "node_id": root.get("_pageindex_node_id"),
                    "title": root.get("title", ""),
                    "summary": root.get("summary", ""),
                    "doc_name": root.get("doc_name", ""),
                }
                for root in roots
                if isinstance(root, dict)
            ],
        }

    def get_document_structure(
        self,
        node_id: Optional[str] = None,
        avoid_chunk_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if node_id is None:
            nodes = self.tree if isinstance(self.tree, list) else [self.tree]
        else:
            node = self.nodes_by_id.get(str(node_id))
            if node is None:
                return []
            nodes = self._get_children(node)

        return [self._node_brief(node, avoid_chunk_ids=avoid_chunk_ids) for node in nodes]

    def get_chunk_content(self, chunk_id: str) -> Dict[str, Any]:
        chunk = self.chunk_lookup.get(str(chunk_id), {})
        return {
            "chunk_id": str(chunk_id),
            "title": chunk.get("title", ""),
            "heading": chunk.get("heading", ""),
            "text": chunk.get("text") or chunk.get("page_content", ""),
            "doc_name": chunk.get("doc_name") or chunk.get("bank_name", ""),
        }

    def _collect_leaf_chunk_ids(self, node: Dict[str, Any]) -> List[str]:
        children = self._get_children(node)
        if not children:
            chunk_id = self._get_chunk_id_from_item(node)
            return [chunk_id] if chunk_id else []

        ids = []
        for child in children:
            ids.extend(self._collect_leaf_chunk_ids(child))
        return ids

    def _is_valid_retrieval_chunk_id(self, chunk_id: Optional[str]) -> bool:
        """
        Return whether a chunk is valid as a retrieval result.

        Intro chunks are intentionally kept as valid retrieval targets because
        they are useful noise / distractor chunks for evaluation.
        Only duplicate chunks are filtered out.
        """
        if not chunk_id:
            return False

        chunk_id = str(chunk_id)

        chunk = self.chunk_lookup.get(chunk_id, {})
        if chunk.get("is_duplicate_chunk", False):
            return False

        return True

    def _branch_has_unseen_chunk(self, node: Dict[str, Any], inspected_chunk_ids: Set[str]) -> bool:
        descendant_chunk_ids = self._collect_leaf_chunk_ids(node)
        return any(
            self._is_valid_retrieval_chunk_id(cid) and cid not in inspected_chunk_ids
            for cid in descendant_chunk_ids
        )

    def _get_unseen_children(self, node: Dict[str, Any], inspected_chunk_ids: Set[str]) -> List[Dict[str, Any]]:
        children = self._get_children(node)
        return [
            child
            for child in children
            if isinstance(child, dict) and self._branch_has_unseen_chunk(child, inspected_chunk_ids)
        ]

    def _node_brief(self, node: Dict[str, Any], avoid_chunk_ids: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Compact node representation shown to the LLM.
        Keeps full summary, but removes unnecessary fields and nested children.
        """
        avoid_chunk_ids = avoid_chunk_ids or set()
        node_id = self._get_node_id_from_node(node)
        is_leaf = self._is_leaf(node)
        chunk_id = self._get_chunk_id_from_item(node) if is_leaf else None
        descendant_chunk_ids = self._collect_leaf_chunk_ids(node)

        unseen_descendants = [
            cid
            for cid in descendant_chunk_ids
            if self._is_valid_retrieval_chunk_id(cid) and cid not in avoid_chunk_ids
        ]

        return {
            "node_id": node_id,
            "title": str(node.get("title") or ""),
            "summary": str(node.get("summary") or ""),
            "is_leaf": is_leaf,
            "chunk_id": chunk_id,
            "num_unseen_descendant_chunks": len(unseen_descendants),
        }

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _call_json_llm(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        estimated_prompt_tokens = sum(
            len(self.encoding.encode(m.get("content", "")))
            for m in messages
        )

        if self.debug_token_counts:
            print(f"Estimated prompt tokens for this LLM call: {estimated_prompt_tokens:,}")

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    max_completion_tokens=2000,
                )

                usage = getattr(response, "usage", None)
                if usage:
                    self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                    self.total_tokens += getattr(usage, "total_tokens", 0) or 0

                content = response.choices[0].message.content
                if content is None or not content.strip():
                    raise ValueError(
                        f"Empty LLM response. Finish reason: {response.choices[0].finish_reason}"
                    )

                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print("\n=== BAD LLM CONTENT START ===")
                    print(repr(content[:1000]))
                    print("=== BAD LLM CONTENT END ===\n")
                    raise

            except Exception as e:
                print(f"LLM call failed, attempt {attempt}/{self.max_retries}: {e}")
                if attempt == self.max_retries:
                    raise
                time.sleep(2 * attempt)

        return {}

    def _select_next_unseen_child_with_llm(
        self,
        query: str,
        current_node: Dict[str, Any],
        children: List[Dict[str, Any]],
        inspected_chunk_ids: Set[str],
    ) -> Optional[Dict[str, Any]]:
        available_children = [
            child
            for child in children
            if self._branch_has_unseen_chunk(child, inspected_chunk_ids)
        ]

        if not available_children:
            return None

        child_options = [
            self._node_brief(child, avoid_chunk_ids=inspected_chunk_ids)
            for child in available_children
        ]

        system_prompt = """
You are a PageIndex retrieval assistant.

You navigate a hierarchical document tree to find chunks relevant to a query.

At each step:
- You receive the current node.
- You receive only its direct child nodes that still contain unseen chunks.
- You must select exactly one child node to explore next.

Use only node title and summary.
Do not answer the query.
Return only valid JSON.

Your entire response must be a single JSON object with exactly these fields:
{
  "selected_node_id": "string",
  "reason": "string"
}
""".strip()

        user_payload = {
            "query": query,
            "current_node": self._node_brief(current_node, avoid_chunk_ids=inspected_chunk_ids),
            "already_inspected_chunk_ids": sorted(list(inspected_chunk_ids)),
            "child_nodes": child_options,
            "instruction": (
                "Select the child node most likely to contain another relevant unseen chunk. "
                "Return only one selected_node_id. Keep reason under 20 words."
            ),
            "output_format": {
                "selected_node_id": "",
                "reason": "",
            },
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        result = self._call_json_llm(messages)
        selected_node_id = str(result.get("selected_node_id", "")).strip()

        for child in available_children:
            child_id = self._get_node_id_from_node(child)
            if child_id == selected_node_id:
                return child

        return available_children[0]


    def _choose_navigation_action_with_llm(
        self,
        query: str,
        current_node: Dict[str, Any],
        children: List[Dict[str, Any]],
        inspected_chunk_ids: Set[str],
        path: List[str],
    ) -> Dict[str, Any]:
        """
        Let the LLM decide whether to:
        - descend into one child
        - backtrack to the parent

        Reading happens automatically when the traversal reaches a leaf.
        """
        available_children = [
            child
            for child in children
            if self._branch_has_unseen_chunk(child, inspected_chunk_ids)
        ]

        child_options = [
            self._node_brief(child, avoid_chunk_ids=inspected_chunk_ids)
            for child in available_children
        ]

        can_backtrack = len(path) > 1

        system_prompt = """
    You are a PageIndex retrieval navigation assistant.

    You navigate a hierarchical document tree to find chunks relevant to a query.

    At each step, you may either:
    1. descend into one child node if a child looks relevant;
    2. backtrack if the current branch does not look useful.

    Only choose backtrack when the visible child nodes do not seem promising for the query.

    Use only node titles and summaries.
    Do not answer the query.
    Return only valid JSON.

    Your entire response must be a single JSON object with exactly these fields:
    {
    "action": "descend or backtrack",
    "selected_node_id": "string or null",
    "reason": "string"
    }
    """.strip()

        user_payload = {
            "query": query,
            "current_path": path,
            "current_node": self._node_brief(
                current_node,
                avoid_chunk_ids=inspected_chunk_ids,
            ),
            "can_backtrack": can_backtrack,
            "already_inspected_chunk_ids": sorted(list(inspected_chunk_ids)),
            "child_nodes": child_options,
            "instruction": (
                "Choose descend if one child is likely to contain a useful chunk. "
                "Choose backtrack if none of the child nodes look useful. "
                "If action is descend, selected_node_id must be one of the child node IDs. "
                "If action is backtrack, selected_node_id must be null. "
                "Keep reason under 20 words."
            ),
            "output_format": {
                "action": "descend",
                "selected_node_id": "",
                "reason": "",
            },
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        result = self._call_json_llm(messages)

        action = str(result.get("action", "")).strip().lower()
        selected_node_id = result.get("selected_node_id")

        if action not in {"descend", "backtrack"}:
            action = "descend"

        if action == "backtrack" and not can_backtrack:
            action = "descend"

        if action == "descend":
            selected_node_id = str(selected_node_id or "").strip()

            for child in available_children:
                child_id = self._get_node_id_from_node(child)
                if child_id == selected_node_id:
                    return {
                        "action": "descend",
                        "selected_child": child,
                        "reason": str(result.get("reason", "")),
                    }

            # fallback if LLM gives invalid child id
            if available_children:
                return {
                    "action": "descend",
                    "selected_child": available_children[0],
                    "reason": "Fallback to first available child.",
                }

            return {
                "action": "backtrack",
                "selected_child": None,
                "reason": "No available children.",
            }

        return {
            "action": "backtrack",
            "selected_child": None,
            "reason": str(result.get("reason", "")),
        }

    def _judge_chunk_sufficiency_with_llm(self, query: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        chunk_id = str(candidate["chunk_id"])
        chunk = self.get_chunk_content(chunk_id)

        system_prompt = """
You are a PageIndex retrieval judge.

You are given a query and one retrieved chunk.
Decide whether this chunk contains sufficient information to answer the query.

Do not answer the query.
Return only valid JSON.

Your entire response must be a single JSON object with exactly these fields:
{
  "is_sufficient": true,
  "reason": "string"
}
""".strip()

        user_payload = {
            "query": query,
            "candidate_chunk": {
                "chunk_id": chunk_id,
                "title": candidate.get("title", "") or chunk.get("title", ""),
                "summary": candidate.get("summary", ""),
                "chunk_excerpt": chunk.get("text", "")[:3500],
            },
            "instruction": (
                "Decide whether this chunk is sufficient to answer the query. "
                "If it is sufficient, retrieval can use this as the top1 candidate. "
                "Keep reason under 25 words."
            ),
            "output_format": {
                "is_sufficient": True,
                "reason": "",
            },
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        result = self._call_json_llm(messages)
        return {
            "is_sufficient": bool(result.get("is_sufficient", False)),
            "reason": str(result.get("reason", "")),
        }

    def _rank_candidate_chunks_with_llm(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidate_payload = []

        for candidate in candidates:
            chunk_id = str(candidate["chunk_id"])
            chunk = self.get_chunk_content(chunk_id)
            candidate_payload.append({
                "chunk_id": chunk_id,
                "node_id": candidate.get("node_id", ""),
                "title": candidate.get("title", "") or chunk.get("title", ""),
                "summary": candidate.get("summary", ""),
                "chunk_excerpt": chunk.get("text", "")[:2500],
            })

        system_prompt = """
You are a PageIndex ranking assistant.

You are given a query and candidate chunks that were inspected through PageIndex retrieval.
Rank the candidate chunks from most relevant to least relevant for answering the query.

Do not answer the query.
Return only valid JSON.

Your entire response must be a single JSON object with exactly this field:
{
  "ranked_chunk_ids": []
}
""".strip()

        user_payload = {
            "query": query,
            "candidate_chunks": candidate_payload,
            "instruction": (
                "Rank the candidate chunks by relevance to the query. "
                "Return only chunk IDs already present in candidate_chunks."
            ),
            "output_format": {
                "ranked_chunk_ids": [],
            },
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        result = self._call_json_llm(messages)
        ranked_ids = [str(cid) for cid in result.get("ranked_chunk_ids", [])]

        candidate_by_id = {str(candidate["chunk_id"]): candidate for candidate in candidates}
        ranked_candidates = []

        for chunk_id in ranked_ids:
            if chunk_id in candidate_by_id:
                ranked_candidates.append(candidate_by_id[chunk_id])

        for candidate in candidates:
            if candidate not in ranked_candidates:
                ranked_candidates.append(candidate)

        return ranked_candidates

    # ------------------------------------------------------------------
    # Stateful traversal
    # ------------------------------------------------------------------

    def inspect_next_chunk_stateful(
        self,
        query: str,
        traversal_state: Dict[str, Any],
        inspected_chunk_ids: Set[str],
        max_steps: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """
        Continue PageIndex traversal from the previous state.
        Does not restart from root for every candidate.
        """
        roots = self.tree if isinstance(self.tree, list) else [self.tree]

        if "synthetic_root" not in traversal_state:
            traversal_state["synthetic_root"] = {
                "_pageindex_node_id": "synthetic_root",
                "node_id": "synthetic_root",
                "title": "Root",
                "summary": "Top-level PageIndex root.",
                "nodes": roots,
            }

        if "stack" not in traversal_state:
            traversal_state["stack"] = [traversal_state["synthetic_root"]]

        if "rejected_child_ids_by_parent" not in traversal_state:
            traversal_state["rejected_child_ids_by_parent"] = {}

        if "navigation_log" not in traversal_state:
            traversal_state["navigation_log"] = []

        stack = traversal_state["stack"]

        for _ in range(max_steps):
            if not stack:
                return None

            current_node = stack[-1]
            current_node_id = self._get_node_id_from_node(current_node)
            children = self._get_children(current_node)

            # Leaf: return valid unseen chunk, then pop it so next call backtracks.
            if not children:
                chunk_id = self._get_chunk_id_from_item(current_node)
                stack.pop()

                if not self._is_valid_retrieval_chunk_id(chunk_id):
                    continue

                if str(chunk_id) in inspected_chunk_ids:
                    continue

                chunk = self.get_chunk_content(str(chunk_id))

                return {
                    "chunk_id": str(chunk_id),
                    "node_id": current_node_id,
                    "title": current_node.get("title", ""),
                    "summary": current_node.get("summary", ""),
                    "path": [self._get_node_id_from_node(node) for node in stack] + [current_node_id],
                    "text": chunk.get("text", ""),
                }

            # If current branch is exhausted, backtrack.
            if not self._branch_has_unseen_chunk(current_node, inspected_chunk_ids):
                stack.pop()
                continue

            raw_unseen_children = self._get_unseen_children(current_node, inspected_chunk_ids)

            rejected_child_ids = traversal_state["rejected_child_ids_by_parent"].get(
                current_node_id,
                set(),
            )

            unseen_children = [
                child
                for child in raw_unseen_children
                if self._get_node_id_from_node(child) not in rejected_child_ids
            ]

            if not unseen_children:
                # This node has no remaining children that are both unseen and not rejected.
                if len(stack) > 1:
                    popped = stack.pop()
                    popped_id = self._get_node_id_from_node(popped)

                    parent_node = stack[-1]
                    parent_id = self._get_node_id_from_node(parent_node)

                    traversal_state["rejected_child_ids_by_parent"].setdefault(
                        parent_id,
                        set(),
                    ).add(popped_id)

                    traversal_state["navigation_log"].append({
                        "step_type": "auto_backtrack",
                        "from_node_id": popped_id,
                        "parent_node_id": parent_id,
                        "reason": "No remaining unseen, non-rejected children.",
                    })

                    print(
                        f"Auto-backtracking from node: {popped_id}. "
                        "No remaining unseen, non-rejected children."
                    )
                    continue

                # At root, there is nowhere else to go.
                return None

            path = [self._get_node_id_from_node(node) for node in stack]

            decision = self._choose_navigation_action_with_llm(
                query=query,
                current_node=current_node,
                children=unseen_children,
                inspected_chunk_ids=inspected_chunk_ids,
                path=path,
            )

            action = decision.get("action")
            reason = decision.get("reason", "")

            if action == "backtrack":
                # Do not pop the synthetic root.
                if len(stack) > 1:
                    popped = stack.pop()
                    popped_id = self._get_node_id_from_node(popped)

                    parent_node = stack[-1]
                    parent_id = self._get_node_id_from_node(parent_node)

                    # Important: remember that this child branch was rejected
                    # under this specific parent.
                    traversal_state["rejected_child_ids_by_parent"].setdefault(
                        parent_id,
                        set(),
                    ).add(popped_id)

                    traversal_state["navigation_log"].append({
                        "step_type": "backtrack",
                        "from_node_id": popped_id,
                        "parent_node_id": parent_id,
                        "reason": reason,
                    })

                    print(f"Backtracking from node: {popped_id}. Reason: {reason}")
                    print(f"Rejected node {popped_id} under parent {parent_id}")
                    time.sleep(self.sleep_between_calls)
                    continue

                # If at root, cannot backtrack; force descent.
                print("At root; cannot backtrack.")

            selected_child = decision.get("selected_child")

            if selected_child is None:
                stack.pop()
                continue

            selected_id = self._get_node_id_from_node(selected_child)
            print(f"Selected next node: {selected_id}. Reason: {reason}")
            
            stack.append(selected_child)
            time.sleep(self.sleep_between_calls)

        return None

    def inspect_next_chunk(
        self,
        query: str,
        avoid_chunk_ids: Optional[Set[str]] = None,
        max_steps: int = 12,
    ) -> Optional[Dict[str, Any]]:
        """
        Backward-compatible non-stateful method.
        Use retrieve_combined() for the improved stateful trajectory.
        """
        traversal_state = {}
        return self.inspect_next_chunk_stateful(
            query=query,
            traversal_state=traversal_state,
            inspected_chunk_ids=avoid_chunk_ids or set(),
            max_steps=max_steps,
        )

    # ------------------------------------------------------------------
    # Retrieval modes
    # ------------------------------------------------------------------

    def retrieve_combined(
        self,
        query: str,
        max_top5_reads: int = 5,
        safety_max_chunk_reads: int = 10,
    ) -> Dict[str, Any]:
        """
        Combined PageIndex retrieval.

        Logic:
        - Inspect chunks one by one using one continuous PageIndex traversal.
        - Stop only when:
            1. a sufficient chunk has been found for top1, AND
            2. at least max_top5_reads chunks have been inspected for top5.
        - If a sufficient chunk is found before 5 chunks, continue until 5 chunks.
        - If 5 chunks are read without a sufficient chunk, rank those 5 for top5,
        but continue reading until a sufficient chunk is found.
        - safety_max_chunk_reads prevents infinite/too expensive retrieval.
        """
        inspected_candidates = []
        inspected_chunk_ids = set()
        first_sufficient_candidate = None
        traversal_state: Dict[str, Any] = {}

        time_at_start = time.time()
        time_at_top1_found = None
        time_at_top5_ready = None

        usage_at_start = self._usage_snapshot()
        usage_at_top1_found = None
        usage_at_top5_ready = None

        attempts = 0

        while attempts < safety_max_chunk_reads:
            # Stop as soon as both objectives are satisfied.
            if (
                first_sufficient_candidate is not None
                and len(inspected_candidates) >= max_top5_reads
            ):
                print(
                    "Stopping: sufficient top1 found and "
                    f"{max_top5_reads} chunks collected for top5."
                )
                break

            attempts += 1

            print(f"\nStateful combined retrieval attempt {attempts}/{safety_max_chunk_reads}")
            print(f"Already inspected: {sorted(inspected_chunk_ids)}")

            candidate = self.inspect_next_chunk_stateful(
                query=query,
                traversal_state=traversal_state,
                inspected_chunk_ids=inspected_chunk_ids,
                max_steps=20,
            )

            if candidate is None:
                print("No candidate returned.")
                break

            chunk_id = str(candidate["chunk_id"])

            if chunk_id in inspected_chunk_ids:
                print(f"Duplicate chunk skipped: {chunk_id}")
                continue

            print(f"Candidate chunk_id: {chunk_id}")

            inspected_chunk_ids.add(chunk_id)
            inspected_candidates.append(candidate)

            # Judge sufficiency only until top1 is found.
            # After top1 is found, we may still continue to 5 chunks,
            # but no more sufficiency calls are needed.
            if first_sufficient_candidate is None:
                judgment = self._judge_chunk_sufficiency_with_llm(
                    query=query,
                    candidate=candidate,
                )

                candidate["sufficiency_judgment"] = judgment
                print(f"Sufficient: {judgment.get('is_sufficient')}")

                if judgment.get("is_sufficient"):
                    first_sufficient_candidate = candidate
                    if usage_at_top1_found is None:
                        usage_at_top1_found = self._usage_snapshot()
                    if time_at_top1_found is None:
                        time_at_top1_found = time.time()
            else:
                candidate["sufficiency_judgment"] = None
                print("Sufficiency already found; skipping sufficiency judge.")

        # ------------------------------------------------------------
        # Rank first 5 inspected chunks for top5.
        # ------------------------------------------------------------
        first_five_candidates = inspected_candidates[:max_top5_reads]

        if first_five_candidates:
            print(f"Ranking first {len(first_five_candidates)} inspected candidates for top5...")
            top5_ranked = self._rank_candidate_chunks_with_llm(
                query=query,
                candidates=first_five_candidates,
            )
            usage_at_top5_ready = self._usage_snapshot()
            time_at_top5_ready = time.time()
        else:
            top5_ranked = []
            usage_at_top5_ready = self._usage_snapshot()
            time_at_top5_ready = time.time()

        # ------------------------------------------------------------
        # Fallback if no sufficient chunk was found.
        # This should only happen if safety_max_chunk_reads is reached
        # or traversal runs out of candidates.
        # ------------------------------------------------------------
        if first_sufficient_candidate is None and inspected_candidates:
            print(
                "No sufficient candidate found before stopping. "
                "Using best ranked inspected candidate as fallback top1."
            )
            fallback_ranked = self._rank_candidate_chunks_with_llm(
                query=query,
                candidates=inspected_candidates,
            )
            first_sufficient_candidate = fallback_ranked[0] if fallback_ranked else None

            if first_sufficient_candidate is not None and usage_at_top1_found is None:
                usage_at_top1_found = self._usage_snapshot()

            if first_sufficient_candidate is not None and time_at_top1_found is None:
                time_at_top1_found = time.time()

        top5_results = self._format_results(top5_ranked[:max_top5_reads])
        top1_results = self._format_results(
            [first_sufficient_candidate] if first_sufficient_candidate else []
        )

        def usage_delta(start, end):
            if start is None or end is None:
                return {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }

            return {
                "prompt_tokens": end["prompt_tokens"] - start["prompt_tokens"],
                "completion_tokens": end["completion_tokens"] - start["completion_tokens"],
                "total_tokens": end["total_tokens"] - start["total_tokens"],
            }
        

        def time_delta(start, end):
            if start is None or end is None:
                return None

            return round(end - start, 3)


        usage_total = usage_delta(usage_at_start, self._usage_snapshot())
        usage_top1 = usage_delta(usage_at_start, usage_at_top1_found)
        usage_top5 = usage_delta(usage_at_start, usage_at_top5_ready)
        latency_top1 = time_delta(time_at_start, time_at_top1_found)
        latency_top5 = time_delta(time_at_start, time_at_top5_ready)
        latency_total = time_delta(time_at_start, time.time())

        return {
            "top1": top1_results,
            "top5": top5_results,
            "usage_top1": usage_top1,
            "usage_top5": usage_top5,
            "usage_total": usage_total,
            "latency_top1_seconds": latency_top1,
            "latency_top5_seconds": latency_top5,
            "latency_total_seconds": latency_total,
            "all_inspected_chunk_ids": [str(c["chunk_id"]) for c in inspected_candidates],
            "num_inspected_chunks": len(inspected_candidates),
            "stopped_because_sufficient_and_top5_ready": (
                first_sufficient_candidate is not None
                and len(inspected_candidates) >= max_top5_reads
            ),
            "navigation_log": traversal_state.get("navigation_log", []),
            "rejected_child_ids_by_parent": {
                parent_id: sorted(list(child_ids))
                for parent_id, child_ids in traversal_state.get("rejected_child_ids_by_parent", {}).items()
            },
        }

    def retrieve_top1(self, query: str, safety_max_chunk_reads: int = 10) -> List[Dict[str, Any]]:
        inspected_candidates = []
        inspected_chunk_ids = set()
        traversal_state: Dict[str, Any] = {}

        attempts = 0

        while len(inspected_candidates) < safety_max_chunk_reads and attempts < safety_max_chunk_reads:
            attempts += 1

            candidate = self.inspect_next_chunk_stateful(
                query=query,
                traversal_state=traversal_state,
                inspected_chunk_ids=inspected_chunk_ids,
                max_steps=20,
            )

            if candidate is None:
                break

            chunk_id = str(candidate["chunk_id"])

            if chunk_id in inspected_chunk_ids:
                continue

            inspected_chunk_ids.add(chunk_id)
            inspected_candidates.append(candidate)

            judgment = self._judge_chunk_sufficiency_with_llm(query=query, candidate=candidate)
            candidate["sufficiency_judgment"] = judgment

            if judgment.get("is_sufficient"):
                return self._format_results([candidate])[:1]

        if not inspected_candidates:
            return []

        ranked = self._rank_candidate_chunks_with_llm(query=query, candidates=inspected_candidates)
        return self._format_results(ranked[:1])

    def retrieve_top5(self, query: str, max_chunk_reads: int = 5) -> List[Dict[str, Any]]:
        inspected_candidates = []
        inspected_chunk_ids = set()
        traversal_state: Dict[str, Any] = {}

        attempts = 0
        max_attempts = max_chunk_reads

        while len(inspected_candidates) < max_chunk_reads and attempts < max_attempts:
            attempts += 1

            print(f"\nTop5 stateful inspection attempt {attempts}/{max_attempts}")
            print(f"Already inspected: {sorted(inspected_chunk_ids)}")

            candidate = self.inspect_next_chunk_stateful(
                query=query,
                traversal_state=traversal_state,
                inspected_chunk_ids=inspected_chunk_ids,
                max_steps=20,
            )

            if candidate is None:
                print("No candidate returned.")
                break

            chunk_id = str(candidate["chunk_id"])
            print(f"Candidate chunk_id: {chunk_id}")

            if chunk_id in inspected_chunk_ids:
                print(f"Duplicate chunk skipped: {chunk_id}")
                continue

            inspected_chunk_ids.add(chunk_id)
            inspected_candidates.append(candidate)

        if not inspected_candidates:
            return []

        ranked = self._rank_candidate_chunks_with_llm(query=query, candidates=inspected_candidates)
        return self._format_results(ranked[:max_chunk_reads])

    def _format_results(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []

        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = str(candidate["chunk_id"])
            chunk = self.chunk_lookup.get(chunk_id, {})

            results.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "source_chunk_id": str(chunk.get("chunk_id", chunk_id)),
                "doc_name": chunk.get("doc_name") or chunk.get("bank_name", ""),
                "node_id": candidate.get("node_id", ""),
                "title": candidate.get("title") or chunk.get("title", ""),
                "summary": candidate.get("summary", ""),
                "score": None,
                "text": chunk.get("text") or chunk.get("page_content", ""),
                "retriever": "pageindex",
                "sufficiency_judgment": candidate.get("sufficiency_judgment"),
            })

        return results

    def get_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }
    
    def _usage_snapshot(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
        }
