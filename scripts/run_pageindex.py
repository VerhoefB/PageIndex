import argparse
import os
import json
import time
import csv
from datetime import datetime
from pageindex import *
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader, reset_llm_usage_tracker, get_llm_usage_tracker


def find_latest_log_file(log_dir, pdf_name):
    if not os.path.isdir(log_dir):
        return ""

    candidates = []
    for filename in os.listdir(log_dir):
        if filename.startswith(pdf_name) and filename.endswith(".json"):
            path = os.path.join(log_dir, filename)
            candidates.append(path)

    if not candidates:
        return ""

    return max(candidates, key=os.path.getmtime)


def extract_run_metadata_from_log(log_path):
    metadata = {
        "num_pages": "",
        "raw_pdf_tokens": "",
        "toc_present": "",
        "toc_page_index_given": "",
    }

    if not log_path or not os.path.isfile(log_path):
        return metadata

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
    except Exception:
        return metadata

    for entry in log_data:
        if not isinstance(entry, dict):
            continue

        if "total_page_number" in entry:
            metadata["num_pages"] = entry.get("total_page_number")

        if "total_token" in entry:
            metadata["raw_pdf_tokens"] = entry.get("total_token")

        if "page_index_given_in_toc" in entry:
            toc_content = entry.get("toc_content")
            toc_page_list = entry.get("toc_page_list") or []

            metadata["toc_present"] = "yes" if toc_content or len(toc_page_list) > 0 else "no"
            metadata["toc_page_index_given"] = entry.get("page_index_given_in_toc")

    return metadata


def append_structure_tracking_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "run_timestamp",
        "dataset",
        "pdf_name",
        "pdf_path",
        "output_path",
        "log_path",
        "status",
        "num_pages",
        "raw_pdf_tokens",
        "toc_present",
        "toc_page_index_given",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "llm_successful_calls",
        "llm_failed_calls",
        "latency_seconds",
        "error",
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF or Markdown document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--md_path', type=str, help='Path to the Markdown file')

    parser.add_argument('--model', type=str, default=None, help='Model to use (overrides config.yaml)')

    parser.add_argument('--toc-check-pages', type=int, default=None,
                      help='Number of pages to check for table of contents (PDF only)')
    parser.add_argument('--max-pages-per-node', type=int, default=None,
                      help='Maximum number of pages per node (PDF only)')
    parser.add_argument('--max-tokens-per-node', type=int, default=None,
                      help='Maximum number of tokens per node (PDF only)')

    parser.add_argument('--if-add-node-id', type=str, default=None,
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default=None,
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default=None,
                      help='Whether to add doc description to the doc')
    parser.add_argument('--if-add-node-text', type=str, default=None,
                      help='Whether to add text to the node')
    
    parser.add_argument('--dataset', type=str, choices=['ESRS', 'FinanceBench'], required=True,
                        help='Dataset name: ESRS or FinanceBench')

    parser.add_argument('--results-root', type=str, default='results',
                        help='Root folder for output structures and tracking CSV')

    parser.add_argument('--data-root', type=str, default='data',
                        help='Root folder for input data and logs')

    parser.add_argument('--tracking-csv', type=str, default=None,
                        help='Path to combined CSV tracking file')
                      
    # Markdown specific arguments
    parser.add_argument('--if-thinning', type=str, default='no',
                      help='Whether to apply tree thinning for markdown (markdown only)')
    parser.add_argument('--thinning-threshold', type=int, default=5000,
                      help='Minimum token threshold for thinning (markdown only)')
    parser.add_argument('--summary-token-threshold', type=int, default=200,
                      help='Token threshold for generating summaries (markdown only)')
    args = parser.parse_args()
    
    # Validate that exactly one file type is specified
    if not args.pdf_path and not args.md_path:
        raise ValueError("Either --pdf_path or --md_path must be specified")
    if args.pdf_path and args.md_path:
        raise ValueError("Only one of --pdf_path or --md_path can be specified")
    
    if args.pdf_path:
        # Validate PDF file
        if not args.pdf_path.lower().endswith('.pdf'):
            raise ValueError("PDF file must have .pdf extension")
        if not os.path.isfile(args.pdf_path):
            raise ValueError(f"PDF file not found: {args.pdf_path}")
            
        # Process PDF file
        user_opt = {
            'model': args.model,
            'toc_check_page_num': args.toc_check_pages,
            'max_page_num_each_node': args.max_pages_per_node,
            'max_token_num_each_node': args.max_tokens_per_node,
            'if_add_node_id': args.if_add_node_id,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
        }
        opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

        pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]

        output_dir = os.path.join(args.results_root, f"{args.dataset} structure")
        output_file = os.path.join(output_dir, f"{pdf_name}_structure.json")
        log_dir = os.path.join(args.data_root, args.dataset, "logs")

        tracking_csv = args.tracking_csv
        if tracking_csv is None:
            tracking_csv = os.path.join(args.results_root, f"{args.dataset}_pageindex_structure_runs.csv")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        reset_llm_usage_tracker()
        start_time = time.perf_counter()
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            toc_with_page_number = page_index_main(args.pdf_path, opt, log_dir=log_dir)

            latency_seconds = time.perf_counter() - start_time
            usage = get_llm_usage_tracker()

            print('Parsing done, saving to file...')

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)

            log_path = find_latest_log_file(log_dir, pdf_name)
            log_metadata = extract_run_metadata_from_log(log_path)

            row = {
                "run_timestamp": run_timestamp,
                "dataset": args.dataset,
                "pdf_name": pdf_name,
                "pdf_path": args.pdf_path,
                "output_path": output_file,
                "log_path": log_path,
                "num_pages": log_metadata["num_pages"],
                "raw_pdf_tokens": log_metadata["raw_pdf_tokens"],
                "toc_present": log_metadata["toc_present"],
                "toc_page_index_given": log_metadata["toc_page_index_given"],
                "status": "success",
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "llm_successful_calls": usage["successful_calls"],
                "llm_failed_calls": usage["failed_calls"],
                "latency_seconds": round(latency_seconds, 3),
                "error": "",
            }

            append_structure_tracking_row(tracking_csv, row)

            print(f'Tree structure saved to: {output_file}')
            print(f'CSV row appended to: {tracking_csv}')

        except Exception as e:
            print(f"PageIndex run failed for {args.pdf_path}")
            print(f"{type(e).__name__}: {e}")
            raise
            
    elif args.md_path:
        # Validate Markdown file
        if not args.md_path.lower().endswith(('.md', '.markdown')):
            raise ValueError("Markdown file must have .md or .markdown extension")
        if not os.path.isfile(args.md_path):
            raise ValueError(f"Markdown file not found: {args.md_path}")
            
        # Process markdown file
        print('Processing markdown file...')
        
        # Process the markdown
        import asyncio
        
        # Use ConfigLoader to get consistent defaults (matching PDF behavior)
        from pageindex.utils import ConfigLoader
        config_loader = ConfigLoader()
        
        # Create options dict with user args
        user_opt = {
            'model': args.model,
            'if_add_node_summary': args.if_add_node_summary,
            'if_add_doc_description': args.if_add_doc_description,
            'if_add_node_text': args.if_add_node_text,
            'if_add_node_id': args.if_add_node_id
        }
        
        # Load config with defaults from config.yaml
        opt = config_loader.load(user_opt)
        
        toc_with_page_number = asyncio.run(md_to_tree(
            md_path=args.md_path,
            if_thinning=args.if_thinning.lower() == 'yes',
            min_token_threshold=args.thinning_threshold,
            if_add_node_summary=opt.if_add_node_summary,
            summary_token_threshold=args.summary_token_threshold,
            model=opt.model,
            if_add_doc_description=opt.if_add_doc_description,
            if_add_node_text=opt.if_add_node_text,
            if_add_node_id=opt.if_add_node_id
        ))
        
        print('Parsing done, saving to file...')
        
        # Save results
        md_name = os.path.splitext(os.path.basename(args.md_path))[0]    
        # output_dir = './results/FinanceBench structure'
        output_dir = './results/ESRS structure'
        output_file = f'{output_dir}/{md_name}_structure.json'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)
        
        print(f'Tree structure saved to: {output_file}')