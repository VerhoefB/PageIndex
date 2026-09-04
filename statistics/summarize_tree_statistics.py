import argparse
import json
from pathlib import Path
from collections import defaultdict
import statistics as stats


M_VALUES = [5, 7, 10]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_root_nodes(data):
    """
    Supports both:
    {
      "doc_name": "...",
      "structure": [...]
    }

    and:
    {
      "nodes": [...]
    }
    """
    if isinstance(data, dict):
        if isinstance(data.get("structure"), list):
            return data["structure"]

        if isinstance(data.get("nodes"), list):
            return data["nodes"]

    if isinstance(data, list):
        return data

    raise ValueError("Could not find root nodes. Expected key 'structure' or 'nodes'.")


def get_doc_name(data, path: Path) -> str:
    if isinstance(data, dict):
        return (
            data.get("doc_name")
            or data.get("bank_name")
            or data.get("doc_id")
            or path.stem.replace("_structure", "")
        )

    return path.stem.replace("_structure", "")


def get_children(node: dict):
    children = node.get("nodes")
    return children if isinstance(children, list) else []


def traverse(nodes, depth=1):
    """
    Traverses one document subtree.
    Depth 1 = root nodes inside that document structure.
    """
    for node in nodes:
        yield node, depth
        children = get_children(node)
        if children:
            yield from traverse(children, depth + 1)


def count_leaf_nodes(nodes) -> int:
    count = 0

    for node, _ in traverse(nodes, depth=1):
        if not get_children(node):
            count += 1

    return count


def summarize_document_subtree(path: Path) -> dict:
    """
    Calculates statistics for one document subtree.
    This is the unit where your hybrid applies m after the document has been selected.
    """

    data = load_json(path)
    roots = get_root_nodes(data)
    doc_name = get_doc_name(data, path)

    nodes_by_depth = defaultdict(list)

    for node, depth in traverse(roots, depth=1):
        nodes_by_depth[depth].append(node)

    level_widths = {
        str(depth): len(nodes)
        for depth, nodes in sorted(nodes_by_depth.items())
    }

    widths = list(level_widths.values())

    total_nodes = sum(widths)
    leaf_nodes = count_leaf_nodes(roots)
    max_depth = max(nodes_by_depth) if nodes_by_depth else 0

    level_retention = {}

    for m in M_VALUES:
        level_retention[str(m)] = {}

        for depth_str, width in level_widths.items():
            retained = min(m, width)

            level_retention[str(m)][depth_str] = {
                "nodes_at_level": width,
                "retained_m": retained,
                "retained_share": round(retained / width, 4) if width else 0,
            }

    return {
        "doc_name": doc_name,
        "file": str(path),

        # Basic subtree statistics
        "total_nodes": total_nodes,
        "leaf_nodes": leaf_nodes,
        "internal_nodes": total_nodes - leaf_nodes,
        "max_depth": max_depth,

        # Width statistics inside this subtree
        "level_widths": level_widths,
        "avg_level_width": round(stats.mean(widths), 2) if widths else 0,
        "median_level_width": stats.median(widths) if widths else 0,
        "min_level_width": min(widths) if widths else 0,
        "max_level_width": max(widths) if widths else 0,

        # m-retention inside this subtree
        "level_retention_for_m": level_retention,
    }


def summarize_full_multidoc_tree(per_document: list[dict]) -> dict:
    """
    Matches hybrid's first step:
    - one artificial full tree;
    - top level contains one node per document;
    - then each document contains its own PageIndex subtree.

    Artificial root itself is not counted.
    Level 1 = document selection level.
    Level 2+ = actual levels inside document subtrees shifted by +1.
    """

    num_documents = len(per_document)

    full_level_widths = defaultdict(int)

    # Level 1 = one document node per document
    full_level_widths[1] = num_documents

    # Shift each document subtree depth by +1
    for doc in per_document:
        for depth_str, width in doc["level_widths"].items():
            full_depth = int(depth_str) + 1
            full_level_widths[full_depth] += width

    full_level_widths = {
        str(depth): width
        for depth, width in sorted(full_level_widths.items())
    }

    widths = list(full_level_widths.values())

    total_document_subtree_nodes = sum(doc["total_nodes"] for doc in per_document)
    total_nodes_including_document_nodes = num_documents + total_document_subtree_nodes
    total_leaf_chunks = sum(doc["leaf_nodes"] for doc in per_document)

    max_depth = max(int(depth) for depth in full_level_widths) if full_level_widths else 0

    full_retention_for_m = {}

    for m in M_VALUES:
        full_retention_for_m[str(m)] = {}

        for depth_str, width in full_level_widths.items():
            retained = min(m, width)

            full_retention_for_m[str(m)][depth_str] = {
                "nodes_at_level": width,
                "retained_m": retained,
                "retained_share": round(retained / width, 4) if width else 0,
            }

    return {
        "tree_type": "full_multidoc_tree",
        "description": (
            "Artificial root is not counted. Level 1 contains one document-level "
            "node per document. Levels 2+ are the document-subtree levels shifted by +1."
        ),

        "num_document_nodes_at_top_level": num_documents,
        "total_nodes_including_document_nodes": total_nodes_including_document_nodes,
        "total_document_subtree_nodes": total_document_subtree_nodes,
        "total_leaf_chunks": total_leaf_chunks,

        "max_depth": max_depth,
        "level_widths": full_level_widths,
        "avg_level_width": round(stats.mean(widths), 2) if widths else 0,
        "median_level_width": stats.median(widths) if widths else 0,
        "min_level_width": min(widths) if widths else 0,
        "max_level_width": max(widths) if widths else 0,

        "level_retention_for_m": full_retention_for_m,
    }


def aggregate_subtree_stats(per_document: list[dict]) -> dict:
    """
    Averages document-subtree statistics across all document subtrees.
    This means each document subtree contributes equally.
    """

    if not per_document:
        return {}

    result = {
        "num_document_subtrees": len(per_document),

        # Nodes per subtree
        "avg_total_nodes_per_document_subtree": round(
            stats.mean(doc["total_nodes"] for doc in per_document), 2
        ),
        "median_total_nodes_per_document_subtree": stats.median(
            doc["total_nodes"] for doc in per_document
        ),
        "min_total_nodes_per_document_subtree": min(
            doc["total_nodes"] for doc in per_document
        ),
        "max_total_nodes_per_document_subtree": max(
            doc["total_nodes"] for doc in per_document
        ),

        # Leaf chunks per subtree
        "avg_leaf_chunks_per_document_subtree": round(
            stats.mean(doc["leaf_nodes"] for doc in per_document), 2
        ),
        "median_leaf_chunks_per_document_subtree": stats.median(
            doc["leaf_nodes"] for doc in per_document
        ),
        "min_leaf_chunks_per_document_subtree": min(
            doc["leaf_nodes"] for doc in per_document
        ),
        "max_leaf_chunks_per_document_subtree": max(
            doc["leaf_nodes"] for doc in per_document
        ),

        # Internal nodes per subtree
        "avg_internal_nodes_per_document_subtree": round(
            stats.mean(doc["internal_nodes"] for doc in per_document), 2
        ),
        "median_internal_nodes_per_document_subtree": stats.median(
            doc["internal_nodes"] for doc in per_document
        ),
        "min_internal_nodes_per_document_subtree": min(
            doc["internal_nodes"] for doc in per_document
        ),
        "max_internal_nodes_per_document_subtree": max(
            doc["internal_nodes"] for doc in per_document
        ),

        # Max depth per subtree
        "avg_max_depth_per_document_subtree": round(
            stats.mean(doc["max_depth"] for doc in per_document), 2
        ),
        "median_max_depth_per_document_subtree": stats.median(
            doc["max_depth"] for doc in per_document
        ),
        "min_max_depth_per_document_subtree": min(
            doc["max_depth"] for doc in per_document
        ),
        "max_max_depth_per_document_subtree": max(
            doc["max_depth"] for doc in per_document
        ),

        # Average level width per subtree
        "avg_avg_level_width_per_document_subtree": round(
            stats.mean(doc["avg_level_width"] for doc in per_document), 2
        ),
        "median_avg_level_width_per_document_subtree": stats.median(
            doc["avg_level_width"] for doc in per_document
        ),
        "min_avg_level_width_per_document_subtree": min(
            doc["avg_level_width"] for doc in per_document
        ),
        "max_avg_level_width_per_document_subtree": max(
            doc["avg_level_width"] for doc in per_document
        ),

        # Median level width per subtree
        "avg_median_level_width_per_document_subtree": round(
            stats.mean(doc["median_level_width"] for doc in per_document), 2
        ),
        "median_median_level_width_per_document_subtree": stats.median(
            doc["median_level_width"] for doc in per_document
        ),

        # Max level width per subtree
        "avg_max_level_width_per_document_subtree": round(
            stats.mean(doc["max_level_width"] for doc in per_document), 2
        ),
        "median_max_level_width_per_document_subtree": stats.median(
            doc["max_level_width"] for doc in per_document
        ),
        "min_max_level_width_per_document_subtree": min(
            doc["max_level_width"] for doc in per_document
        ),
        "max_max_level_width_per_document_subtree": max(
            doc["max_level_width"] for doc in per_document
        ),
    }

    # Average retained share per subtree for m = 5, 7, 10.
    # First calculate the average retained share inside each subtree,
    # then average those subtree-level values across documents.
    for m in M_VALUES:
        subtree_avg_retained_shares = []

        for doc in per_document:
            widths = [
                int(width)
                for width in doc["level_widths"].values()
                if int(width) > 0
            ]

            if widths:
                subtree_avg_retained_share = stats.mean(
                    min(m, width) / width
                    for width in widths
                )
                subtree_avg_retained_shares.append(subtree_avg_retained_share)

        result[f"avg_retained_share_across_subtrees_m_{m}"] = round(
            stats.mean(subtree_avg_retained_shares), 4
        ) if subtree_avg_retained_shares else 0

        result[f"median_retained_share_across_subtrees_m_{m}"] = round(
            stats.median(subtree_avg_retained_shares), 4
        ) if subtree_avg_retained_shares else 0

    return result


def aggregate_level_widths_across_subtrees(per_document: list[dict]) -> dict:
    """
    For each level inside document subtrees:
    - collect that level's width from every document that has that level;
    - then calculate average, median, min, max.

    This is useful because your m is applied level-wise inside the selected subtree.
    """

    widths_by_level = defaultdict(list)

    for doc in per_document:
        for depth_str, width in doc["level_widths"].items():
            widths_by_level[int(depth_str)].append(int(width))

    result = {}

    for depth, widths in sorted(widths_by_level.items()):
        entry = {
            "num_documents_with_this_level": len(widths),
            "avg_width": round(stats.mean(widths), 2),
            "median_width": stats.median(widths),
            "min_width": min(widths),
            "max_width": max(widths),
        }

        for m in M_VALUES:
            retained_shares = [
                min(m, width) / width
                for width in widths
                if width > 0
            ]

            entry[f"avg_retained_share_m_{m}"] = round(
                stats.mean(retained_shares), 4
            ) if retained_shares else 0

            entry[f"median_retained_share_m_{m}"] = round(
                stats.median(retained_shares), 4
            ) if retained_shares else 0

        result[str(depth)] = entry

    return result


def build_thesis_table(output: dict) -> dict:
    """
    Creates a compact table-like dictionary with the most useful thesis numbers.
    """

    full = output["full_multidoc_tree"]
    agg = output["aggregate_document_subtrees"]

    table = {
        "Full multi-document tree": {
            "Top-level document nodes": full["num_document_nodes_at_top_level"],
            "Total nodes including document nodes": full["total_nodes_including_document_nodes"],
            "Total document-subtree nodes": full["total_document_subtree_nodes"],
            "Total leaf chunks": full["total_leaf_chunks"],
            "Max depth": full["max_depth"],
            "Max level width": full["max_level_width"],
        },
        "Average over document subtrees": {
            "Average total nodes per subtree": agg["avg_total_nodes_per_document_subtree"],
            "Median total nodes per subtree": agg["median_total_nodes_per_document_subtree"],
            "Average leaf chunks per subtree": agg["avg_leaf_chunks_per_document_subtree"],
            "Median leaf chunks per subtree": agg["median_leaf_chunks_per_document_subtree"],
            "Average max depth per subtree": agg["avg_max_depth_per_document_subtree"],
            "Median max depth per subtree": agg["median_max_depth_per_document_subtree"],
            "Average max level width per subtree": agg["avg_max_level_width_per_document_subtree"],
            "Median max level width per subtree": agg["median_max_level_width_per_document_subtree"],
            "Average retained share m=5": agg["avg_retained_share_across_subtrees_m_5"],
            "Average retained share m=7": agg["avg_retained_share_across_subtrees_m_7"],
            "Average retained share m=10": agg["avg_retained_share_across_subtrees_m_10"],
        },
    }

    return table


def print_summary(output: dict):
    full = output["full_multidoc_tree"]
    agg = output["aggregate_document_subtrees"]
    levels = output["average_level_widths_across_document_subtrees"]

    print("\n=== HYBRID TREE STRUCTURE SUMMARY ===")

    print("\n--- Full multi-document tree ---")
    print(f"Top-level document nodes: {full['num_document_nodes_at_top_level']}")
    print(f"Total nodes including document nodes: {full['total_nodes_including_document_nodes']}")
    print(f"Total document-subtree nodes: {full['total_document_subtree_nodes']}")
    print(f"Total leaf chunks: {full['total_leaf_chunks']}")
    print(f"Max depth: {full['max_depth']}")
    print(f"Average level width: {full['avg_level_width']}")
    print(f"Median level width: {full['median_level_width']}")
    print(f"Max level width: {full['max_level_width']}")
    print(f"Level widths: {full['level_widths']}")

    print("\n--- Average over document subtrees ---")
    print(f"Number of document subtrees: {agg['num_document_subtrees']}")

    print(f"Average total nodes per subtree: {agg['avg_total_nodes_per_document_subtree']}")
    print(f"Median total nodes per subtree: {agg['median_total_nodes_per_document_subtree']}")
    print(f"Min total nodes per subtree: {agg['min_total_nodes_per_document_subtree']}")
    print(f"Max total nodes per subtree: {agg['max_total_nodes_per_document_subtree']}")

    print(f"Average leaf chunks per subtree: {agg['avg_leaf_chunks_per_document_subtree']}")
    print(f"Median leaf chunks per subtree: {agg['median_leaf_chunks_per_document_subtree']}")
    print(f"Min leaf chunks per subtree: {agg['min_leaf_chunks_per_document_subtree']}")
    print(f"Max leaf chunks per subtree: {agg['max_leaf_chunks_per_document_subtree']}")

    print(f"Average internal nodes per subtree: {agg['avg_internal_nodes_per_document_subtree']}")
    print(f"Median internal nodes per subtree: {agg['median_internal_nodes_per_document_subtree']}")

    print(f"Average max depth per subtree: {agg['avg_max_depth_per_document_subtree']}")
    print(f"Median max depth per subtree: {agg['median_max_depth_per_document_subtree']}")
    print(f"Min max depth per subtree: {agg['min_max_depth_per_document_subtree']}")
    print(f"Max max depth per subtree: {agg['max_max_depth_per_document_subtree']}")

    print(f"Average of average level width per subtree: {agg['avg_avg_level_width_per_document_subtree']}")
    print(f"Median of average level width per subtree: {agg['median_avg_level_width_per_document_subtree']}")

    print(f"Average max level width per subtree: {agg['avg_max_level_width_per_document_subtree']}")
    print(f"Median max level width per subtree: {agg['median_max_level_width_per_document_subtree']}")
    print(f"Min max level width per subtree: {agg['min_max_level_width_per_document_subtree']}")
    print(f"Max max level width per subtree: {agg['max_max_level_width_per_document_subtree']}")

    for m in M_VALUES:
        print(
            f"Average retained share across subtrees for m={m}: "
            f"{agg[f'avg_retained_share_across_subtrees_m_{m}']}"
        )

    print("\n--- Average level widths inside document subtrees ---")
    for depth, values in levels.items():
        print(
            f"Subtree level {depth}: "
            f"avg width={values['avg_width']}, "
            f"median width={values['median_width']}, "
            f"min width={values['min_width']}, "
            f"max width={values['max_width']}, "
            f"avg retained share m=5={values['avg_retained_share_m_5']}, "
            f"m=7={values['avg_retained_share_m_7']}, "
            f"m=10={values['avg_retained_share_m_10']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize full multi-document tree and document-subtree statistics for hybrid PageIndex."
    )

    parser.add_argument(
        "--structure-dir",
        required=True,
        help="Directory containing *_structure_new.json files."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file for all statistics."
    )

    parser.add_argument(
        "--thesis-table-output",
        default=None,
        help="Optional compact JSON file with thesis-ready table numbers."
    )

    args = parser.parse_args()

    structure_dir = Path(args.structure_dir)

    files = sorted(
        path for path in structure_dir.glob("*_structure_new.json")
        if "combined" not in path.name.lower()
    )

    if not files:
        raise FileNotFoundError(f"No *_structure_new.json files found in {structure_dir}")

    per_document = [summarize_document_subtree(path) for path in files]

    output = {
        "full_multidoc_tree": summarize_full_multidoc_tree(per_document),
        "aggregate_document_subtrees": aggregate_subtree_stats(per_document),
        "average_level_widths_across_document_subtrees": aggregate_level_widths_across_subtrees(per_document),
        "per_document_subtrees": per_document,
    }

    output["thesis_table"] = build_thesis_table(output)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if args.thesis_table_output:
        thesis_table_path = Path(args.thesis_table_output)
        thesis_table_path.parent.mkdir(parents=True, exist_ok=True)

        with open(thesis_table_path, "w", encoding="utf-8") as f:
            json.dump(output["thesis_table"], f, indent=2, ensure_ascii=False)

    print_summary(output)

    print(f"\nSaved full statistics to: {output_path}")

    if args.thesis_table_output:
        print(f"Saved thesis table to: {args.thesis_table_output}")


if __name__ == "__main__":
    main()