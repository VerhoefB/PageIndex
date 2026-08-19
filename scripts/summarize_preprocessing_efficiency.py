import argparse
from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "num_pages",
    "raw_pdf_tokens",
    "num_chunks",
    "total_chunk_tokens",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "llm_successful_calls",
    "llm_failed_calls",
    "latency_seconds",
]


def normalize_name(value):
    if pd.isna(value):
        return ""

    return str(value).strip().replace("\\", "/")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    if "run_timestamp" in df.columns:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"], errors="coerce")

    return df


def keep_latest_run_per_document(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps the latest run for each dataset/pdf_name combination.
    This avoids double-counting if a document was processed multiple times.
    """

    if "run_timestamp" not in df.columns:
        return df.drop_duplicates(subset=["dataset", "pdf_name"], keep="last")

    return (
        df.sort_values("run_timestamp")
        .drop_duplicates(subset=["dataset", "pdf_name"], keep="last")
        .reset_index(drop=True)
    )


def clean_structure_runs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "status" in df.columns:
        df = df[df["status"].fillna("").str.lower() == "success"].copy()

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["pdf_name"] = df["pdf_name"].apply(normalize_name)

    df["toc_present"] = df.get("toc_present", "unknown")
    df["toc_page_index_given"] = df.get("toc_page_index_given", "unknown")

    df["toc_present"] = df["toc_present"].astype(str).str.lower().str.strip()
    df["toc_page_index_given"] = df["toc_page_index_given"].astype(str).str.lower().str.strip()

    df = keep_latest_run_per_document(df)

    rename_map = {
        "input_tokens": "structure_input_tokens",
        "output_tokens": "structure_output_tokens",
        "total_tokens": "structure_total_tokens",
        "llm_successful_calls": "structure_llm_successful_calls",
        "llm_failed_calls": "structure_llm_failed_calls",
        "latency_seconds": "structure_latency_seconds",
    }

    df = df.rename(columns=rename_map)

    return df


def clean_chunk_runs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["pdf_name"] = df["pdf_name"].apply(normalize_name)

    df = keep_latest_run_per_document(df)

    rename_map = {
        "input_tokens": "chunk_input_tokens",
        "output_tokens": "chunk_output_tokens",
        "total_tokens": "chunk_total_tokens",
        "llm_successful_calls": "chunk_llm_successful_calls",
        "llm_failed_calls": "chunk_llm_failed_calls",
        "latency_seconds": "chunk_latency_seconds",
    }

    df = df.rename(columns=rename_map)

    keep_columns = [
        "dataset",
        "pdf_name",
        "doc_name",
        "num_chunks",
        "total_chunk_tokens",
        "chunk_input_tokens",
        "chunk_output_tokens",
        "chunk_total_tokens",
        "chunk_llm_successful_calls",
        "chunk_llm_failed_calls",
        "chunk_latency_seconds",
    ]

    keep_columns = [col for col in keep_columns if col in df.columns]

    return df[keep_columns]


def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, pd.NA)
    return numerator / denominator


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in [
        "structure_input_tokens",
        "structure_output_tokens",
        "structure_total_tokens",
        "structure_llm_successful_calls",
        "structure_llm_failed_calls",
        "structure_latency_seconds",
        "chunk_input_tokens",
        "chunk_output_tokens",
        "chunk_total_tokens",
        "chunk_llm_successful_calls",
        "chunk_llm_failed_calls",
        "chunk_latency_seconds",
        "num_pages",
        "raw_pdf_tokens",
        "num_chunks",
        "total_chunk_tokens",
    ]:
        if column not in df.columns:
            df[column] = 0

        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["combined_input_tokens"] = df["structure_input_tokens"] + df["chunk_input_tokens"]
    df["combined_output_tokens"] = df["structure_output_tokens"] + df["chunk_output_tokens"]
    df["combined_total_tokens"] = df["structure_total_tokens"] + df["chunk_total_tokens"]

    df["combined_llm_successful_calls"] = (
        df["structure_llm_successful_calls"] + df["chunk_llm_successful_calls"]
    )
    df["combined_llm_failed_calls"] = (
        df["structure_llm_failed_calls"] + df["chunk_llm_failed_calls"]
    )
    df["combined_latency_seconds"] = (
        df["structure_latency_seconds"] + df["chunk_latency_seconds"]
    )

    # Structure construction per-page metrics
    df["structure_total_tokens_per_page"] = safe_divide(
        df["structure_total_tokens"], df["num_pages"]
    )
    df["structure_latency_seconds_per_page"] = safe_divide(
        df["structure_latency_seconds"], df["num_pages"]
    )
    df["structure_llm_calls_per_page"] = safe_divide(
        df["structure_llm_successful_calls"], df["num_pages"]
    )

    # Chunk construction per-page and per-chunk metrics
    df["chunk_total_tokens_per_page"] = safe_divide(
        df["chunk_total_tokens"], df["num_pages"]
    )
    df["chunk_latency_seconds_per_page"] = safe_divide(
        df["chunk_latency_seconds"], df["num_pages"]
    )
    df["chunk_total_tokens_per_chunk"] = safe_divide(
        df["chunk_total_tokens"], df["num_chunks"]
    )
    df["chunk_latency_seconds_per_chunk"] = safe_divide(
        df["chunk_latency_seconds"], df["num_chunks"]
    )

    # Combined preprocessing metrics
    df["combined_total_tokens_per_page"] = safe_divide(
        df["combined_total_tokens"], df["num_pages"]
    )
    df["combined_latency_seconds_per_page"] = safe_divide(
        df["combined_latency_seconds"], df["num_pages"]
    )
    df["combined_total_tokens_per_chunk"] = safe_divide(
        df["combined_total_tokens"], df["num_chunks"]
    )
    df["combined_latency_seconds_per_chunk"] = safe_divide(
        df["combined_latency_seconds"], df["num_chunks"]
    )

    return df


def summarize_group(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = {
        "pdf_name": "count",
        "num_pages": "sum",
        "num_chunks": "sum",
        "raw_pdf_tokens": "sum",
        "total_chunk_tokens": "sum",

        "structure_input_tokens": "sum",
        "structure_output_tokens": "sum",
        "structure_total_tokens": "sum",
        "structure_llm_successful_calls": "sum",
        "structure_llm_failed_calls": "sum",
        "structure_latency_seconds": "sum",

        "chunk_input_tokens": "sum",
        "chunk_output_tokens": "sum",
        "chunk_total_tokens": "sum",
        "chunk_llm_successful_calls": "sum",
        "chunk_llm_failed_calls": "sum",
        "chunk_latency_seconds": "sum",

        "combined_input_tokens": "sum",
        "combined_output_tokens": "sum",
        "combined_total_tokens": "sum",
        "combined_llm_successful_calls": "sum",
        "combined_llm_failed_calls": "sum",
        "combined_latency_seconds": "sum",

        "structure_total_tokens_per_page": "mean",
        "structure_latency_seconds_per_page": "mean",
        "chunk_total_tokens_per_page": "mean",
        "chunk_latency_seconds_per_page": "mean",
        "combined_total_tokens_per_page": "mean",
        "combined_latency_seconds_per_page": "mean",

        "chunk_total_tokens_per_chunk": "mean",
        "chunk_latency_seconds_per_chunk": "mean",
        "combined_total_tokens_per_chunk": "mean",
        "combined_latency_seconds_per_chunk": "mean",
    }

    available_metrics = {
        key: value for key, value in metrics.items()
        if key in df.columns
    }

    summary = (
        df.groupby(group_columns, dropna=False)
        .agg(available_metrics)
        .reset_index()
    )

    summary = summary.rename(columns={"pdf_name": "num_documents"})

    # Dataset-level weighted per-page/per-chunk metrics
    summary["structure_total_tokens_per_page_weighted"] = (
        summary["structure_total_tokens"] / summary["num_pages"].replace(0, pd.NA)
    )
    summary["structure_latency_seconds_per_page_weighted"] = (
        summary["structure_latency_seconds"] / summary["num_pages"].replace(0, pd.NA)
    )

    summary["chunk_total_tokens_per_page_weighted"] = (
        summary["chunk_total_tokens"] / summary["num_pages"].replace(0, pd.NA)
    )
    summary["chunk_latency_seconds_per_page_weighted"] = (
        summary["chunk_latency_seconds"] / summary["num_pages"].replace(0, pd.NA)
    )

    summary["combined_total_tokens_per_page_weighted"] = (
        summary["combined_total_tokens"] / summary["num_pages"].replace(0, pd.NA)
    )
    summary["combined_latency_seconds_per_page_weighted"] = (
        summary["combined_latency_seconds"] / summary["num_pages"].replace(0, pd.NA)
    )

    summary["chunk_total_tokens_per_chunk_weighted"] = (
        summary["chunk_total_tokens"] / summary["num_chunks"].replace(0, pd.NA)
    )
    summary["chunk_latency_seconds_per_chunk_weighted"] = (
        summary["chunk_latency_seconds"] / summary["num_chunks"].replace(0, pd.NA)
    )

    summary["combined_total_tokens_per_chunk_weighted"] = (
        summary["combined_total_tokens"] / summary["num_chunks"].replace(0, pd.NA)
    )
    summary["combined_latency_seconds_per_chunk_weighted"] = (
        summary["combined_latency_seconds"] / summary["num_chunks"].replace(0, pd.NA)
    )

    return summary.round(4)


def build_stage_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, group in df.groupby("dataset", dropna=False):
        rows.append({
            "dataset": dataset,
            "stage": "tree_construction",
            "num_documents": len(group),
            "num_pages": group["num_pages"].sum(),
            "num_chunks": group["num_chunks"].sum(),
            "input_tokens": group["structure_input_tokens"].sum(),
            "output_tokens": group["structure_output_tokens"].sum(),
            "total_tokens": group["structure_total_tokens"].sum(),
            "successful_llm_calls": group["structure_llm_successful_calls"].sum(),
            "failed_llm_calls": group["structure_llm_failed_calls"].sum(),
            "latency_seconds": group["structure_latency_seconds"].sum(),
            "tokens_per_page": group["structure_total_tokens"].sum() / group["num_pages"].sum(),
            "latency_seconds_per_page": group["structure_latency_seconds"].sum() / group["num_pages"].sum(),
        })

        rows.append({
            "dataset": dataset,
            "stage": "chunk_construction",
            "num_documents": len(group),
            "num_pages": group["num_pages"].sum(),
            "num_chunks": group["num_chunks"].sum(),
            "input_tokens": group["chunk_input_tokens"].sum(),
            "output_tokens": group["chunk_output_tokens"].sum(),
            "total_tokens": group["chunk_total_tokens"].sum(),
            "successful_llm_calls": group["chunk_llm_successful_calls"].sum(),
            "failed_llm_calls": group["chunk_llm_failed_calls"].sum(),
            "latency_seconds": group["chunk_latency_seconds"].sum(),
            "tokens_per_page": group["chunk_total_tokens"].sum() / group["num_pages"].sum(),
            "latency_seconds_per_page": group["chunk_latency_seconds"].sum() / group["num_pages"].sum(),
            "tokens_per_chunk": group["chunk_total_tokens"].sum() / group["num_chunks"].sum(),
            "latency_seconds_per_chunk": group["chunk_latency_seconds"].sum() / group["num_chunks"].sum(),
        })

        rows.append({
            "dataset": dataset,
            "stage": "combined_preprocessing",
            "num_documents": len(group),
            "num_pages": group["num_pages"].sum(),
            "num_chunks": group["num_chunks"].sum(),
            "input_tokens": group["combined_input_tokens"].sum(),
            "output_tokens": group["combined_output_tokens"].sum(),
            "total_tokens": group["combined_total_tokens"].sum(),
            "successful_llm_calls": group["combined_llm_successful_calls"].sum(),
            "failed_llm_calls": group["combined_llm_failed_calls"].sum(),
            "latency_seconds": group["combined_latency_seconds"].sum(),
            "tokens_per_page": group["combined_total_tokens"].sum() / group["num_pages"].sum(),
            "latency_seconds_per_page": group["combined_latency_seconds"].sum() / group["num_pages"].sum(),
            "tokens_per_chunk": group["combined_total_tokens"].sum() / group["num_chunks"].sum(),
            "latency_seconds_per_chunk": group["combined_latency_seconds"].sum() / group["num_chunks"].sum(),
        })

    return pd.DataFrame(rows).round(4)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize PageIndex tree-construction and chunk-construction efficiency."
    )

    parser.add_argument(
        "--structure-runs",
        required=True,
        help="CSV file with PageIndex structure construction runs."
    )

    parser.add_argument(
        "--chunk-runs",
        required=True,
        help="CSV file with chunk construction runs."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output summaries are written."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. ESRS or FinanceBench."
    )

    args = parser.parse_args()

    structure_runs = clean_structure_runs(load_csv(Path(args.structure_runs)))
    chunk_runs = clean_chunk_runs(load_csv(Path(args.chunk_runs)))

    structure_runs["dataset"] = args.dataset
    chunk_runs["dataset"] = args.dataset

    merged = structure_runs.merge(
        chunk_runs,
        on=["dataset", "pdf_name"],
        how="left",
        suffixes=("_structure", "_chunk"),
    )

    merged = add_derived_metrics(merged)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_output = output_dir / f"{args.dataset}_preprocessing_efficiency_per_document.csv"
    toc_summary_output = output_dir / f"{args.dataset}_preprocessing_efficiency_by_toc.csv"
    stage_summary_output = output_dir / f"{args.dataset}_preprocessing_efficiency_by_stage.csv"

    merged.round(4).to_csv(detailed_output, index=False)

    toc_summary = summarize_group(
        merged,
        ["dataset", "toc_present", "toc_page_index_given"]
    )
    toc_summary.to_csv(toc_summary_output, index=False)

    stage_summary = build_stage_summary(merged)
    stage_summary.to_csv(stage_summary_output, index=False)

    print(f"\nSaved outputs for {args.dataset}:")
    print(f"  {detailed_output}")
    print(f"  {toc_summary_output}")
    print(f"  {stage_summary_output}")

    print(f"\nStage summary for {args.dataset}:")
    print(stage_summary.to_string(index=False))


if __name__ == "__main__":
    main()