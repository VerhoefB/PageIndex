import argparse
import copy
import json
import os
import re
from glob import glob


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_doc_id(name):
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r"_structure$", "", name)
    name = re.sub(r"_chunks$", "", name)
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name

def clean_document_title(value):
    value = str(value or "").strip()

    if not value:
        return ""

    value = value.replace("_", " ")
    value = value.replace("-", " ")

    # Collapse repeated whitespace
    value = " ".join(value.split())

    return value

def prefix_tree_ids(nodes, doc_id, doc_name):
    for node in nodes:
        node["doc_id"] = doc_id
        node["doc_name"] = doc_name

        if node.get("node_id"):
            node["local_node_id"] = str(node["node_id"])
            node["node_id"] = f"{doc_id}_{node['node_id']}"

        if node.get("parent_node_id"):
            node["local_parent_node_id"] = str(node["parent_node_id"])
            node["parent_node_id"] = f"{doc_id}_{node['parent_node_id']}"

        if node.get("chunk_id"):
            node["local_chunk_id"] = str(node["chunk_id"])
            node["chunk_id"] = f"{doc_id}_{node['chunk_id']}"

        if node.get("canonical_chunk_id"):
            node["local_canonical_chunk_id"] = str(node["canonical_chunk_id"])
            node["canonical_chunk_id"] = f"{doc_id}_{node['canonical_chunk_id']}"

        if node.get("retrieval_chunk_id"):
            node["local_retrieval_chunk_id"] = str(node["retrieval_chunk_id"])
            node["retrieval_chunk_id"] = f"{doc_id}_{node['retrieval_chunk_id']}"

        if node.get("source_chunk_id"):
            node["local_source_chunk_id"] = str(node["source_chunk_id"])
            node["source_chunk_id"] = f"{doc_id}_{node['source_chunk_id']}"

        if node.get("nodes"):
            prefix_tree_ids(node["nodes"], doc_id, doc_name)


def prefix_chunk(row, doc_id, doc_name, source_pdf=None):
    row = dict(row)

    row["doc_id"] = doc_id
    row["doc_name"] = doc_name

    if source_pdf:
        row["source_pdf"] = source_pdf

    # Preserve local IDs for traceability.
    row["local_chunk_id"] = str(row.get("chunk_id", ""))
    row["local_node_id"] = str(row.get("node_id", ""))
    row["local_parent_node_id"] = str(row.get("parent_node_id", ""))

    if row.get("canonical_chunk_id"):
        row["local_canonical_chunk_id"] = str(row.get("canonical_chunk_id"))

    if row.get("retrieval_chunk_id"):
        row["local_retrieval_chunk_id"] = str(row.get("retrieval_chunk_id"))

    if row.get("duplicate_of_chunk_id"):
        row["local_duplicate_of_chunk_id"] = str(row.get("duplicate_of_chunk_id"))

    # Prefix main IDs.
    if row.get("chunk_id"):
        row["chunk_id"] = f"{doc_id}_{row['chunk_id']}"

    if row.get("node_id"):
        row["node_id"] = f"{doc_id}_{row['node_id']}"

    if row.get("parent_node_id"):
        row["parent_node_id"] = f"{doc_id}_{row['parent_node_id']}"

    # Prefix canonical/retrieval duplicate fields too.
    if row.get("canonical_chunk_id"):
        row["canonical_chunk_id"] = f"{doc_id}_{row['canonical_chunk_id']}"

    if row.get("retrieval_chunk_id"):
        row["retrieval_chunk_id"] = f"{doc_id}_{row['retrieval_chunk_id']}"

    if row.get("duplicate_of_chunk_id"):
        row["duplicate_of_chunk_id"] = f"{doc_id}_{row['duplicate_of_chunk_id']}"

    return row


def combine(structure_dir, chunks_dir, output_structure, output_chunks, collection_name):
    structure_files = sorted(glob(os.path.join(structure_dir, "*_structure_new.json")))

    collection = {
        "title": collection_name,
        "node_id": "collection",
        "node_type": "collection_root",
        "nodes": []
    }

    all_chunks = []

    for structure_path in structure_files:
        base = os.path.basename(structure_path).replace("_structure_new.json", "")
        chunks_path = os.path.join(chunks_dir, f"{base}_chunks.jsonl")

        if not os.path.exists(chunks_path):
            print(f"Skipping {base}: no chunks file found")
            continue

        structure_data = load_json(structure_path)
        chunks = load_jsonl(chunks_path)

        doc_id = clean_doc_id(base)
        doc_name = structure_data.get("doc_name") or base
        doc_name = os.path.splitext(doc_name)[0]

        structure = copy.deepcopy(structure_data.get("structure", []))
        prefix_tree_ids(structure, doc_id, doc_name)

        doc_node = {
            "title": clean_document_title(doc_name),
            "summary": structure_data.get("doc_description", ""),
            "node_id": doc_id,
            "node_type": "document_node",
            "doc_id": doc_id,
            "doc_name": doc_name,
            "source_structure_file": structure_path,
            "source_chunks_file": chunks_path,
            "nodes": structure,
        }

        collection["nodes"].append(doc_node)

        for row in chunks:
            all_chunks.append(
                prefix_chunk(
                    row,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    source_pdf=structure_data.get("source_pdf", "")
                )
            )

        print(f"Added {doc_name}: {len(chunks)} chunks")

    combined_structure = {
        "doc_name": collection_name,
        "structure": [collection],
    }

    save_json(output_structure, combined_structure)
    save_jsonl(output_chunks, all_chunks)

    print(f"\nDocuments: {len(collection['nodes'])}")
    print(f"Chunks: {len(all_chunks)}")
    print(f"Saved structure: {output_structure}")
    print(f"Saved chunks: {output_chunks}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure_dir", required=True)
    parser.add_argument("--chunks_dir", required=True)
    parser.add_argument("--output_structure", required=True)
    parser.add_argument("--output_chunks", required=True)
    parser.add_argument("--collection_name", default="ESRS bank reports")

    args = parser.parse_args()

    combine(
        structure_dir=args.structure_dir,
        chunks_dir=args.chunks_dir,
        output_structure=args.output_structure,
        output_chunks=args.output_chunks,
        collection_name=args.collection_name,
    )