import argparse
import csv
import json
import os
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_structure(tree_json):
    if isinstance(tree_json, dict) and "structure" in tree_json:
        return tree_json["structure"]
    return tree_json


def flatten_nodes(nodes, path=None, rows=None):
    """
    Flatten the tree in the exact order used by the structure.
    This is preorder: parent first, then children.
    """
    if path is None:
        path = []
    if rows is None:
        rows = []

    for node in nodes:
        node_id = node.get("node_id", "")
        title = node.get("title", "")
        structure = node.get("structure", "")

        current_path = path + [f"{structure} {title}".strip()]

        rows.append({
            "node_id": node_id,
            "structure": structure,
            "title": title,
            "heading": node.get("heading", ""),
            "start_index": node.get("start_index"),
            "end_index": node.get("end_index"),
            "path": " > ".join(current_path),
            "has_children": bool(node.get("nodes")),
        })

        if node.get("nodes"):
            flatten_nodes(node["nodes"], current_path, rows)

    return rows


def flatten_leaf_nodes(nodes, path=None, rows=None):
    """
    Flatten only leaf nodes, because these become chunks.
    """
    if path is None:
        path = []
    if rows is None:
        rows = []

    for node in nodes:
        title = node.get("title", "")
        structure = node.get("structure", "")
        current_path = path + [f"{structure} {title}".strip()]

        if node.get("nodes"):
            flatten_leaf_nodes(node["nodes"], current_path, rows)
        else:
            rows.append({
                "node_id": node.get("node_id", ""),
                "structure": structure,
                "title": title,
                "heading": node.get("heading", ""),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
                "path": " > ".join(current_path),
            })

    return rows


def safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def validate_order(rows, doc_name, check_name):
    """
    Checks:
    1. Missing start/end indexes.
    2. start_index > end_index.
    3. Current start_index goes backwards compared to previous node start.
    4. Current start_index goes backwards compared to previous node end.
    """
    issues = []

    previous_start = None
    previous_end = None
    previous_row = None

    for position, row in enumerate(rows):
        start = safe_int(row.get("start_index"))
        end = safe_int(row.get("end_index"))

        if start is None or end is None:
            issues.append({
                "doc_name": doc_name,
                "check": check_name,
                "issue_type": "missing_index",
                "position": position,
                "node_id": row.get("node_id", ""),
                "structure": row.get("structure", ""),
                "title": row.get("title", ""),
                "start_index": row.get("start_index"),
                "end_index": row.get("end_index"),
                "previous_node_id": previous_row.get("node_id", "") if previous_row else "",
                "previous_structure": previous_row.get("structure", "") if previous_row else "",
                "previous_title": previous_row.get("title", "") if previous_row else "",
                "previous_start_index": previous_start,
                "previous_end_index": previous_end,
                "path": row.get("path", ""),
                "message": "Node has missing or invalid start_index/end_index.",
            })
            continue

        if start > end:
            issues.append({
                "doc_name": doc_name,
                "check": check_name,
                "issue_type": "start_after_end",
                "position": position,
                "node_id": row.get("node_id", ""),
                "structure": row.get("structure", ""),
                "title": row.get("title", ""),
                "start_index": start,
                "end_index": end,
                "previous_node_id": previous_row.get("node_id", "") if previous_row else "",
                "previous_structure": previous_row.get("structure", "") if previous_row else "",
                "previous_title": previous_row.get("title", "") if previous_row else "",
                "previous_start_index": previous_start,
                "previous_end_index": previous_end,
                "path": row.get("path", ""),
                "message": "Node start_index is larger than end_index.",
            })

        if previous_start is not None and start < previous_start:
            issues.append({
                "doc_name": doc_name,
                "check": check_name,
                "issue_type": "start_index_goes_backwards",
                "position": position,
                "node_id": row.get("node_id", ""),
                "structure": row.get("structure", ""),
                "title": row.get("title", ""),
                "start_index": start,
                "end_index": end,
                "previous_node_id": previous_row.get("node_id", "") if previous_row else "",
                "previous_structure": previous_row.get("structure", "") if previous_row else "",
                "previous_title": previous_row.get("title", "") if previous_row else "",
                "previous_start_index": previous_start,
                "previous_end_index": previous_end,
                "path": row.get("path", ""),
                "message": "Node start_index is smaller than the previous node start_index.",
            })

        if previous_end is not None and start < previous_end:
            issues.append({
                "doc_name": doc_name,
                "check": check_name,
                "issue_type": "start_before_previous_end",
                "position": position,
                "node_id": row.get("node_id", ""),
                "structure": row.get("structure", ""),
                "title": row.get("title", ""),
                "start_index": start,
                "end_index": end,
                "previous_node_id": previous_row.get("node_id", "") if previous_row else "",
                "previous_structure": previous_row.get("structure", "") if previous_row else "",
                "previous_title": previous_row.get("title", "") if previous_row else "",
                "previous_start_index": previous_start,
                "previous_end_index": previous_end,
                "path": row.get("path", ""),
                "message": "Node start_index is smaller than the previous node end_index.",
            })

        previous_start = start
        previous_end = end
        previous_row = row

    return issues


def validate_sibling_order(nodes, doc_name, parent_path=""):
    """
    Checks ordering only among siblings.

    This is useful because child nodes are allowed to be within the same
    page range as their parent, but siblings should usually move forward.
    """
    issues = []

    previous_start = None
    previous_end = None
    previous_node = None

    for position, node in enumerate(nodes):
        start = safe_int(node.get("start_index"))
        end = safe_int(node.get("end_index"))

        node_path = f"{parent_path} > {node.get('structure', '')} {node.get('title', '')}".strip(" >")

        if start is not None and end is not None:
            if start > end:
                issues.append({
                    "doc_name": doc_name,
                    "check": "sibling_order",
                    "issue_type": "start_after_end",
                    "position": position,
                    "node_id": node.get("node_id", ""),
                    "structure": node.get("structure", ""),
                    "title": node.get("title", ""),
                    "start_index": start,
                    "end_index": end,
                    "previous_node_id": previous_node.get("node_id", "") if previous_node else "",
                    "previous_structure": previous_node.get("structure", "") if previous_node else "",
                    "previous_title": previous_node.get("title", "") if previous_node else "",
                    "previous_start_index": previous_start,
                    "previous_end_index": previous_end,
                    "path": node_path,
                    "message": "Sibling node start_index is larger than end_index.",
                })

            if previous_start is not None and start < previous_start:
                issues.append({
                    "doc_name": doc_name,
                    "check": "sibling_order",
                    "issue_type": "sibling_start_goes_backwards",
                    "position": position,
                    "node_id": node.get("node_id", ""),
                    "structure": node.get("structure", ""),
                    "title": node.get("title", ""),
                    "start_index": start,
                    "end_index": end,
                    "previous_node_id": previous_node.get("node_id", "") if previous_node else "",
                    "previous_structure": previous_node.get("structure", "") if previous_node else "",
                    "previous_title": previous_node.get("title", "") if previous_node else "",
                    "previous_start_index": previous_start,
                    "previous_end_index": previous_end,
                    "path": node_path,
                    "message": "Sibling start_index goes backwards compared with previous sibling.",
                })

            if previous_end is not None and start < previous_end:
                issues.append({
                    "doc_name": doc_name,
                    "check": "sibling_order",
                    "issue_type": "sibling_start_before_previous_end",
                    "position": position,
                    "node_id": node.get("node_id", ""),
                    "structure": node.get("structure", ""),
                    "title": node.get("title", ""),
                    "start_index": start,
                    "end_index": end,
                    "previous_node_id": previous_node.get("node_id", "") if previous_node else "",
                    "previous_structure": previous_node.get("structure", "") if previous_node else "",
                    "previous_title": previous_node.get("title", "") if previous_node else "",
                    "previous_start_index": previous_start,
                    "previous_end_index": previous_end,
                    "path": node_path,
                    "message": "Sibling start_index is smaller than previous sibling end_index.",
                })

            previous_start = start
            previous_end = end
            previous_node = node

        if node.get("nodes"):
            issues.extend(
                validate_sibling_order(
                    node["nodes"],
                    doc_name=doc_name,
                    parent_path=node_path,
                )
            )

    return issues


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if not rows:
        rows = [{
            "doc_name": "",
            "check": "",
            "issue_type": "",
            "position": "",
            "node_id": "",
            "structure": "",
            "title": "",
            "start_index": "",
            "end_index": "",
            "previous_node_id": "",
            "previous_structure": "",
            "previous_title": "",
            "previous_start_index": "",
            "previous_end_index": "",
            "path": "",
            "message": "No issues found.",
        }]

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_file(structure_path, output_path=None):
    tree_json = load_json(structure_path)
    structure = get_structure(tree_json)

    doc_name = (
        tree_json.get("doc_name")
        if isinstance(tree_json, dict)
        else Path(structure_path).stem
    )

    all_nodes = flatten_nodes(structure)
    leaf_nodes = flatten_leaf_nodes(structure)

    issues = []

    # All nodes in preorder: good for seeing total structural weirdness.
    issues.extend(
        validate_order(
            rows=all_nodes,
            doc_name=doc_name,
            check_name="all_nodes_preorder",
        )
    )

    # Leaf nodes only: closest to actual chunk sequence.
    issues.extend(
        validate_order(
            rows=leaf_nodes,
            doc_name=doc_name,
            check_name="leaf_nodes_order",
        )
    )

    # Siblings only: avoids over-flagging parent-child overlap.
    issues.extend(
        validate_sibling_order(
            nodes=structure,
            doc_name=doc_name,
        )
    )

    if output_path is not None:
        write_csv(issues, output_path)

    print("\n=== STRUCTURE INDEX VALIDATION ===")
    print(f"Structure: {structure_path}")
    print(f"Document: {doc_name}")
    print(f"All nodes: {len(all_nodes)}")
    print(f"Leaf nodes: {len(leaf_nodes)}")
    print(f"Issues: {len(issues)}")

    if output_path is not None:
        print(f"Report: {output_path}")

    if issues:
        print("\nFirst 10 issues:")
        for issue in issues[:10]:
            print(
                f"- [{issue['check']}] {issue['issue_type']} | "
                f"{issue['previous_structure']} {issue['previous_title']} "
                f"({issue['previous_start_index']}-{issue['previous_end_index']}) "
                f"→ {issue['structure']} {issue['title']} "
                f"({issue['start_index']}-{issue['end_index']})"
            )


def validate_directory(structure_dir, output_path):
    all_issues = []

    structure_paths = sorted(
        Path(structure_dir).glob("*_structure.json")
    )

    # Avoid accidentally validating combined structures here.
    structure_paths = [
        path for path in structure_paths
        if "combined" not in path.name.lower()
        and not path.name.endswith("_structure_new.json")
    ]

    for structure_path in structure_paths:
        tree_json = load_json(structure_path)
        structure = get_structure(tree_json)

        doc_name = (
            tree_json.get("doc_name")
            if isinstance(tree_json, dict)
            else structure_path.stem
        )

        all_nodes = flatten_nodes(structure)
        leaf_nodes = flatten_leaf_nodes(structure)

        all_issues.extend(
            validate_order(
                rows=all_nodes,
                doc_name=doc_name,
                check_name="all_nodes_preorder",
            )
        )

        all_issues.extend(
            validate_order(
                rows=leaf_nodes,
                doc_name=doc_name,
                check_name="leaf_nodes_order",
            )
        )

        all_issues.extend(
            validate_sibling_order(
                nodes=structure,
                doc_name=doc_name,
            )
        )

    write_csv(all_issues, output_path)

    print("\n=== STRUCTURE DIRECTORY INDEX VALIDATION ===")
    print(f"Directory: {structure_dir}")
    print(f"Files checked: {len(structure_paths)}")
    print(f"Issues: {len(all_issues)}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--structure", default=None)
    parser.add_argument("--structure-dir", default=None)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.structure:
        validate_file(
            structure_path=args.structure,
            output_path=args.output,
        )

    elif args.structure_dir:
        validate_directory(
            structure_dir=args.structure_dir,
            output_path=args.output,
        )

    else:
        raise ValueError("Provide either --structure or --structure-dir.")