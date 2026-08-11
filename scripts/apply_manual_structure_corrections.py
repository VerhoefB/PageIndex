import argparse
import copy
import json
from pathlib import Path


def get_children(node):
    return node.get("nodes", []) or []


def iter_nodes(nodes):
    for node in nodes:
        yield node
        yield from iter_nodes(get_children(node))


def find_node(nodes, node_id):
    for node in iter_nodes(nodes):
        if str(node.get("node_id")) == str(node_id):
            return node
    return None


def find_node_with_parent(nodes, node_id, parent=None):
    """
    Returns:
    node, parent_node, siblings_list, index_in_siblings

    parent_node is None for top-level nodes.
    """
    for i, node in enumerate(nodes):
        if str(node.get("node_id")) == str(node_id):
            return node, parent, nodes, i

        result = find_node_with_parent(get_children(node), node_id, node)
        if result is not None:
            return result

    return None


def remove_node(nodes, node_id):
    result = find_node_with_parent(nodes, node_id)
    if result is None:
        raise ValueError(f"Cannot remove node_id={node_id}: node not found.")

    node, parent, siblings, index = result
    removed = siblings.pop(index)
    return removed


def insert_node(nodes, node, target_parent_node_id, position, reference_node_id=None):
    """
    target_parent_node_id:
    - "__root__" = insert directly in the top-level structure list
    - None = insert at top level for first/last,
             or insert in same sibling list as reference node for before/after
    - otherwise insert inside that parent node's children.

    position:
      - first
      - last
      - before
      - after
    """

    # Explicit root-level before/after insertion
    if target_parent_node_id == "__root__" and position in {"before", "after"}:
        if reference_node_id is None:
            raise ValueError(f"Position '{position}' requires reference_node_id.")

        for i, child in enumerate(nodes):
            if str(child.get("node_id")) == str(reference_node_id):
                insert_at = i if position == "before" else i + 1
                nodes.insert(insert_at, node)
                return

        raise ValueError(
            f"Cannot insert node_id={node.get('node_id')}: "
            f"reference_node_id={reference_node_id} not found at root level."
        )

    if position in {"before", "after"}:
        if reference_node_id is None:
            raise ValueError(f"Position '{position}' requires reference_node_id.")

        # If a target parent is given, search only inside that parent's children.
        if target_parent_node_id is not None:
            target_parent = find_node(nodes, target_parent_node_id)
            if target_parent is None:
                raise ValueError(
                    f"Cannot insert node_id={node.get('node_id')}: "
                    f"target_parent_node_id={target_parent_node_id} not found."
                )

            target_children = target_parent.get("nodes", [])
            target_parent["nodes"] = target_children

            for i, child in enumerate(target_children):
                if str(child.get("node_id")) == str(reference_node_id):
                    insert_at = i if position == "before" else i + 1
                    target_children.insert(insert_at, node)
                    return

            raise ValueError(
                f"Cannot insert node_id={node.get('node_id')}: "
                f"reference_node_id={reference_node_id} not found under target parent."
            )

        # If no target parent is given, find the reference node anywhere
        # and insert the moved node into the same siblings list.
        reference_result = find_node_with_parent(nodes, reference_node_id)
        if reference_result is None:
            raise ValueError(
                f"Cannot insert node_id={node.get('node_id')}: "
                f"reference_node_id={reference_node_id} not found anywhere in the tree."
            )

        reference_node, reference_parent, reference_siblings, reference_index = reference_result
        insert_at = reference_index if position == "before" else reference_index + 1
        reference_siblings.insert(insert_at, node)
        return

    # first / last insertion
    if target_parent_node_id is None or target_parent_node_id == "__root__":
        target_children = nodes
    else:
        target_parent = find_node(nodes, target_parent_node_id)
        if target_parent is None:
            raise ValueError(
                f"Cannot insert node_id={node.get('node_id')}: "
                f"target_parent_node_id={target_parent_node_id} not found."
            )
        target_children = target_parent.get("nodes", [])
        target_parent["nodes"] = target_children

    if position == "first":
        target_children.insert(0, node)
        return

    if position == "last":
        target_children.append(node)
        return

    raise ValueError(f"Unknown position: {position}")


def move_node(nodes, node_id, target_parent_node_id, position, reference_node_id=None):
    node = remove_node(nodes, node_id)

    insert_node(
        nodes=nodes,
        node=node,
        target_parent_node_id=target_parent_node_id,
        position=position,
        reference_node_id=reference_node_id,
    )

    return node


def update_indexes(nodes, node_id, start_index, end_index):
    node = find_node(nodes, node_id)
    if node is None:
        raise ValueError(f"Cannot update indexes for node_id={node_id}: node not found.")

    old_start = node.get("start_index")
    old_end = node.get("end_index")

    node["start_index"] = start_index
    node["end_index"] = end_index

    return {
        "node_id": node_id,
        "old_start_index": old_start,
        "old_end_index": old_end,
        "new_start_index": start_index,
        "new_end_index": end_index,
    }


def apply_corrections(structure_data, corrections):
    cleaned = copy.deepcopy(structure_data)

    if "structure" not in cleaned or not isinstance(cleaned["structure"], list):
        raise ValueError("Input structure JSON must contain a top-level list field named 'structure'.")

    nodes = cleaned["structure"]
    audit_log = []

    for op_number, operation in enumerate(corrections.get("operations", []), start=1):
        action = operation.get("action")
        node_id = operation.get("node_id")
        reason = operation.get("reason", "")

        if not action:
            raise ValueError(f"Operation {op_number} has no action.")
        if not node_id:
            raise ValueError(f"Operation {op_number} has no node_id.")

        if action == "remove":
            removed = remove_node(nodes, node_id)

            audit_log.append({
                "operation": op_number,
                "action": "remove",
                "node_id": node_id,
                "title": removed.get("title"),
                "start_index": removed.get("start_index"),
                "end_index": removed.get("end_index"),
                "reason": reason,
            })

        elif action == "move":
            moved = move_node(
                nodes=nodes,
                node_id=node_id,
                target_parent_node_id=operation.get("target_parent_node_id"),
                position=operation.get("position", "last"),
                reference_node_id=operation.get("reference_node_id"),
            )

            audit_log.append({
                "operation": op_number,
                "action": "move",
                "node_id": node_id,
                "title": moved.get("title"),
                "target_parent_node_id": operation.get("target_parent_node_id"),
                "position": operation.get("position", "last"),
                "reference_node_id": operation.get("reference_node_id"),
                "reason": reason,
            })

        elif action == "update_indexes":
            change = update_indexes(
                nodes=nodes,
                node_id=node_id,
                start_index=operation["start_index"],
                end_index=operation["end_index"],
            )

            change.update({
                "operation": op_number,
                "action": "update_indexes",
                "reason": reason,
            })
            audit_log.append(change)

        else:
            raise ValueError(f"Unknown action '{action}' in operation {op_number}.")

    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", required=True, help="Path to original PageIndex structure JSON.")
    parser.add_argument("--corrections", required=True, help="Path to manual corrections JSON.")
    parser.add_argument("--output", required=True, help="Path to cleaned output structure JSON.")
    args = parser.parse_args()

    structure_path = Path(args.structure)
    corrections_path = Path(args.corrections)
    output_path = Path(args.output)

    with structure_path.open("r", encoding="utf-8") as f:
        structure_data = json.load(f)

    with corrections_path.open("r", encoding="utf-8") as f:
        corrections = json.load(f)

    cleaned = apply_corrections(structure_data, corrections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Saved cleaned structure to: {output_path}")
    print(f"Applied {len(corrections.get('operations', []))} correction operations.")


if __name__ == "__main__":
    main()