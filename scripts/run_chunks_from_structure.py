import argparse
import csv
import json
import os
import time
from datetime import datetime

from pageindex.page_index import build_chunks_from_existing_structure
from pageindex.leaf_text_store_pageaware import save_leaf_text_jsonl
from pageindex.utils import (
    ConfigLoader,
    reset_llm_usage_tracker,
    get_llm_usage_tracker,
)


def append_chunk_tracking_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "run_timestamp",
        "dataset",
        "pdf_name",
        "doc_name",
        "pdf_path",
        "structure_path",
        "output_path",
        "updated_structure_path",
        "num_chunks",
        "total_chunk_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "llm_successful_calls",
        "llm_failed_calls",
        "latency_seconds",
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_doc_name_from_structure(structure_path):
    with open(structure_path, "r", encoding="utf-8") as f:
        structure_data = json.load(f)

    return structure_data.get("doc_name") or os.path.splitext(os.path.basename(structure_path))[0]


def add_dataset_metadata_to_chunks(leaf_text_rows, dataset, doc_name):
    for row in leaf_text_rows:
        row["doc_name"] = doc_name

        if dataset == "ESRS":
            row["bank_name"] = doc_name

            # Do not overwrite these.
            # The chunking code already decides per chunk.
            row.setdefault("generate_queries", False)
            row.setdefault("skip_query_reason", "")
            row.setdefault("num_queries", 0)

        elif dataset == "FinanceBench":
            row.pop("bank_name", None)
            row["generate_queries"] = False
            row["skip_query_reason"] = "FinanceBench uses existing benchmark questions."
            row["num_queries"] = 0

    return leaf_text_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--structure_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updated_structure_output", default=None)

    parser.add_argument("--dataset", type=str, choices=["ESRS", "FinanceBench"], required=True)

    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--max-tokens-per-node", type=int, default=None)

    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--tracking-csv", type=str, default=None)

    parser.add_argument("--num-queries", type=int, default=10)

    args = parser.parse_args()

    opt = ConfigLoader().load({
        k: v for k, v in {
            "model": args.model,
            "max_token_num_each_node": args.max_tokens_per_node,
        }.items()
        if v is not None
    })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.updated_structure_output:
        os.makedirs(os.path.dirname(args.updated_structure_output), exist_ok=True)

    tracking_csv = args.tracking_csv
    if tracking_csv is None:
        tracking_csv = os.path.join(args.results_root, f"{args.dataset}_chunk_runs.csv")

    pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
    doc_name = load_doc_name_from_structure(args.structure_path)

    reset_llm_usage_tracker()
    start_time = time.perf_counter()
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        leaf_text_rows = build_chunks_from_existing_structure(
            pdf_path=args.pdf_path,
            structure_path=args.structure_path,
            output_path=args.output,
            opt=opt,
            updated_structure_path=args.updated_structure_output,
        )

        leaf_text_rows = add_dataset_metadata_to_chunks(
            leaf_text_rows=leaf_text_rows,
            dataset=args.dataset,
            doc_name=doc_name,
        )

        # Save again after adding dataset-specific metadata.
        save_leaf_text_jsonl(leaf_text_rows, args.output)

        latency_seconds = time.perf_counter() - start_time
        usage = get_llm_usage_tracker()

        num_chunks = len(leaf_text_rows)
        total_chunk_tokens = sum(
            int(row.get("token_count") or row.get("text_token_count") or 0)
            for row in leaf_text_rows
        )

        row = {
            "run_timestamp": run_timestamp,
            "dataset": args.dataset,
            "pdf_name": pdf_name,
            "doc_name": doc_name,
            "pdf_path": args.pdf_path,
            "structure_path": args.structure_path,
            "output_path": args.output,
            "updated_structure_path": args.updated_structure_output or "",
            "num_chunks": num_chunks,
            "total_chunk_tokens": total_chunk_tokens,
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "llm_successful_calls": usage["successful_calls"],
            "llm_failed_calls": usage["failed_calls"],
            "latency_seconds": round(latency_seconds, 3),
        }

        append_chunk_tracking_row(tracking_csv, row)

        print(f"Chunks saved to: {args.output}")
        print(f"Updated structure saved to: {args.updated_structure_output}")
        print(f"Chunk tracking CSV updated: {tracking_csv}")
        print(f"Chunks: {num_chunks}")
        print(f"Latency: {latency_seconds:.2f}s")
        print(f"LLM usage: {usage}")

    except Exception as e:
        print(f"Chunking failed for {args.pdf_path}")
        print(f"{type(e).__name__}: {e}")
        raise