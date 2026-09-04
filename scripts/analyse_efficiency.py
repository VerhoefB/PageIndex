import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paths

FINAL_RESULTS = Path("final results")

ANALYSIS_PATHS = {
    "ESRS": Path("final results/ESRS results analysis/query_level_combined.csv"),
    "FinanceBench": Path("final results/FinanceBench results analysis/query_level_combined.csv"),
}

STRUCTURE_PATHS = {
    "ESRS": Path("final results/ESRS_pageindex_structure_runs.csv"),
    "FinanceBench": Path("final results/FinanceBench_pageindex_structure_runs.csv"),
}

OUTPUT_DIR = Path("Efficiency results analysis")


# General helpers

def clean_string(value):
    if value is None:
        return ""

    value = str(value)

    if value.lower() in {"nan", "none"}:
        return ""

    return " ".join(value.strip().split())


def to_bool(value):
    return clean_string(value).lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def pretty_model(model):
    low = clean_string(model).lower()

    if "harrier" in low:
        return "Harrier"

    if "linq" in low and "mistral" in low:
        return "Linq-Embed-Mistral"

    if "gpt-5" in low:
        return "GPT-5"

    return clean_string(model)


def make_query_key(row, dataset):
    """
    Create a query key that is consistent across retrieval methods.
    """

    financebench_id = clean_string(row.get("financebench_id", ""))

    if dataset == "FinanceBench" and financebench_id:
        return f"FB::{financebench_id}"

    doc_name = clean_string(
        row.get("doc_name", row.get("bank_name", ""))
    ).casefold()

    ground_truth_chunk_id = clean_string(
        row.get("ground_truth_chunk_id", "")
    ).casefold()

    query = clean_string(row.get("query", "")).casefold()

    return (
        f"ESRS::{doc_name}::"
        f"{ground_truth_chunk_id}::"
        f"{query}"
    )


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# Robust setup-CSV reader

def read_valid_csv_rows(path):
    """
    Read only setup rows that match the original CSV header.
    Later runs contain additional cache fields.
    """

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    header = rows[0]

    output = []

    for row in rows[1:]:

        if len(row) != len(header):
            continue

        output.append(dict(zip(header, row)))

    return output


# Find setup CSVs automatically

def find_setup_rows():
    dense_rows = []
    hybrid_rows = []

    for path in FINAL_RESULTS.rglob("*.csv"):

        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
        except Exception:
            continue

        header_set = set(header)

        # Dense setup CSV
        if {
            "embedding_seconds",
            "model_load_seconds",
            "cache_loaded",
            "total_setup_seconds",
        }.issubset(header_set):

            for row in read_valid_csv_rows(path):
                row["_source_file"] = str(path)
                dense_rows.append(row)

        # Hybrid setup CSV
        elif {
            "node_embedding_seconds",
            "node_cache_loaded",
            "top_node_cache_loaded",
            "model_and_index_seconds",
            "total_setup_seconds",
        }.issubset(header_set):

            for row in read_valid_csv_rows(path):
                row["_source_file"] = str(path)
                hybrid_rows.append(row)

    return pd.DataFrame(dense_rows), pd.DataFrame(hybrid_rows)


# PageIndex tree preprocessing

def load_structure_preprocessing(dataset):
    path = STRUCTURE_PATHS[dataset]

    if not path.exists():
        raise FileNotFoundError(
            f"Structure tracking file not found: {path}"
        )

    df = pd.read_csv(path)

    if "status" in df.columns:
        df = df[
            ~df["status"].astype(str).str.lower().isin(
                ["failed", "error"]
            )
        ].copy()

    # Original tree-construction run per document.
    # Later correction/post-processing runs must not be counted again.
    if "run_timestamp" in df.columns:
        df = df.sort_values("run_timestamp")

    df = df.drop_duplicates("pdf_name", keep="first")

    numeric_cols = [
        "num_pages",
        "raw_pdf_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_seconds",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return {
        "documents": df["pdf_name"].nunique(),
        "pages": df["num_pages"].sum(),
        "input_tokens": df["input_tokens"].sum(),
        "output_tokens": df["output_tokens"].sum(),
        "total_tokens": df["total_tokens"].sum(),
        "latency_seconds": df["latency_seconds"].sum(),
    }


# BM25 preprocessing

def find_bm25_result(dataset):
    candidates = list(
        FINAL_RESULTS.rglob("bm25_query_results.jsonl")
    )

    dataset_low = dataset.lower()

    candidates = [
        p for p in candidates
        if dataset_low in str(p).lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find BM25 result file for {dataset}"
        )

    # Prefer the most direct path if copies exist
    candidates = sorted(
        candidates,
        key=lambda p: len(str(p))
    )

    return candidates[0]


def load_bm25_preprocessing(dataset):
    path = find_bm25_result(dataset)
    rows = read_jsonl(path)

    if not rows:
        return None

    row = rows[0]

    return {
        "preprocessing_seconds": float(
            row.get("preprocessing_time_seconds", np.nan)
        ),
        "tokenization_seconds": float(
            row.get("tokenization_time_seconds", np.nan)
        ),
        "index_construction_seconds": float(
            row.get("index_construction_time_seconds", np.nan)
        ),
        "num_chunks_indexed": float(
            row.get("num_chunks_indexed", np.nan)
        ),
    }


# Dense preprocessing

def select_dense_uncached_rows(dense_setup, dataset):
    if dense_setup.empty:
        return pd.DataFrame()

    df = dense_setup.copy()

    df = df[
        df["dataset"].astype(str) == dataset
    ].copy()

    df["cache_loaded_bool"] = df["cache_loaded"].map(to_bool)

    # Only actual corpus-embedding runs
    df = df[~df["cache_loaded_bool"]].copy()

    df["model_pretty"] = df["model"].map(pretty_model)

    numeric_cols = [
        "model_load_seconds",
        "embedding_seconds",
        "index_build_seconds",
        "total_setup_seconds",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # One original uncached run per model
    df = (
        df.sort_values("run_timestamp")
        .drop_duplicates("model_pretty", keep="first")
    )

    return df


# Hybrid preprocessing

def select_hybrid_setup_rows(hybrid_setup, dataset):
    if hybrid_setup.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = hybrid_setup.copy()

    df = df[
        df["dataset"].astype(str) == dataset
    ].copy()

    df["model_pretty"] = df["model"].map(pretty_model)

    df["node_cache_bool"] = df["node_cache_loaded"].map(to_bool)
    df["top_node_cache_bool"] = df[
        "top_node_cache_loaded"
    ].map(to_bool)

    df["total_setup_seconds"] = pd.to_numeric(
        df["total_setup_seconds"],
        errors="coerce",
    )

    df["model_and_index_seconds"] = pd.to_numeric(
        df["model_and_index_seconds"],
        errors="coerce",
    )

    # Original uncached node embedding runs

    node_runs = df[
        (~df["node_cache_bool"])
        & (~df["top_node_cache_bool"])
    ].copy()

    node_runs = (
        node_runs.sort_values("run_timestamp")
        .drop_duplicates("model_pretty", keep="first")
    )

    # Original chunk-reranking embedding runs
    #
    # The node caches already existed when the chunk embeddings
    # were first created, so the old setup logs do not identify
    # these runs separately. They are identified by their much
    # larger setup time.

    cached_runs = df[
        df["node_cache_bool"]
        & df["top_node_cache_bool"]
    ].copy()

    chunk_runs = []

    for model, group in cached_runs.groupby("model_pretty"):

        group = group.dropna(
            subset=["model_and_index_seconds"]
        )

        if group.empty:
            continue

        # The run that created all chunk embeddings is
        # orders of magnitude larger than normal cache loading.
        largest = group.loc[
            group["model_and_index_seconds"].idxmax()
        ].copy()

        largest["model_pretty"] = model

        chunk_runs.append(largest)

    chunk_runs = pd.DataFrame(chunk_runs)

    return node_runs, chunk_runs


# Preprocessing output

def build_preprocessing_table(dense_setup, hybrid_setup):
    output = []

    for dataset in ["ESRS", "FinanceBench"]:

        # BM25

        bm25 = load_bm25_preprocessing(dataset)

        output.append({
            "dataset": dataset,
            "method": "BM25",
            "model": "–",
            "component": "Tokenization + BM25 index construction",
            "latency_seconds": bm25["preprocessing_seconds"],
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "notes": (
                f"Tokenization={bm25['tokenization_seconds']:.3f}s; "
                f"index construction={bm25['index_construction_seconds']:.3f}s"
            ),
        })

        # Dense

        dense = select_dense_uncached_rows(
            dense_setup,
            dataset,
        )

        for _, row in dense.iterrows():

            output.append({
                "dataset": dataset,
                "method": "Dense",
                "model": row["model_pretty"],
                "component": "Model load + chunk embedding / FAISS setup",
                "latency_seconds": row["total_setup_seconds"],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "notes": (
                    f"Model load={row['model_load_seconds']:.3f}s; "
                    f"embedding/index stage={row['embedding_seconds']:.3f}s"
                ),
            })

        # PageIndex tree construction

        structure = load_structure_preprocessing(dataset)

        output.append({
            "dataset": dataset,
            "method": "PageIndex",
            "model": "GPT-5",
            "component": "Hierarchical tree construction",
            "latency_seconds": structure["latency_seconds"],
            "input_tokens": structure["input_tokens"],
            "output_tokens": structure["output_tokens"],
            "total_tokens": structure["total_tokens"],
            "notes": (
                f"{structure['documents']} documents; "
                f"{int(structure['pages']):,} pages. "
                "The same PageIndex tree is reused by Hybrid PageIndex."
            ),
        })

        # Hybrid

        node_runs, chunk_runs = select_hybrid_setup_rows(
            hybrid_setup,
            dataset,
        )

        for _, row in node_runs.iterrows():

            output.append({
                "dataset": dataset,
                "method": "Hybrid PageIndex",
                "model": row["model_pretty"],
                "component": "Node + top-document embedding setup",
                "latency_seconds": row["total_setup_seconds"],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "notes": (
                    "Embeds hierarchy nodes; PageIndex tree "
                    "construction reported separately."
                ),
            })

        for _, row in chunk_runs.iterrows():

            output.append({
                "dataset": dataset,
                "method": "Hybrid PageIndex",
                "model": row["model_pretty"],
                "component": "Full-chunk reranker embedding setup",
                "latency_seconds": row["model_and_index_seconds"],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "notes": (
                    "Observed first chunk-cache creation run. "
                    "Includes small model/cache loading overhead; "
                    "chunk representation is title + heading + text."
                ),
            })

    return pd.DataFrame(output)


# Query embedding timing files

def discover_query_embedding_timings():
    files = list(
        Path(".").rglob("*query_embedding_timings.csv")
    )

    frames = []

    for path in files:

        low = path.name.lower()

        if "esrs" in low:
            dataset = "ESRS"
        elif "financebench" in low:
            dataset = "FinanceBench"
        else:
            continue

        if "harrier" in low:
            model = "Harrier"
        elif "linq" in low:
            model = "Linq-Embed-Mistral"
        else:
            continue

        df = pd.read_csv(path)

        if "embedding_latency_seconds" not in df.columns:
            continue

        df["dataset"] = dataset
        df["model"] = model

        df["query_key"] = df.apply(
            lambda row: make_query_key(row, dataset),
            axis=1,
        )

        df["embedding_latency_seconds"] = pd.to_numeric(
            df["embedding_latency_seconds"],
            errors="coerce",
        )

        frames.append(
            df[
                [
                    "dataset",
                    "model",
                    "query_key",
                    "embedding_latency_seconds",
                ]
            ]
        )

        print(
            f"Loaded query timings: "
            f"{dataset} / {model} / {len(df)} queries"
        )

    if not frames:
        raise FileNotFoundError(
            "No *query_embedding_timings.csv files found."
        )

    timings = pd.concat(
        frames,
        ignore_index=True,
    )

    timings = timings.drop_duplicates(
        ["dataset", "model", "query_key"]
    )

    return timings


# Query-time efficiency

def load_query_results():
    frames = []

    for dataset, path in ANALYSIS_PATHS.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Could not find: {path}"
            )

        df = pd.read_csv(path)

        df["dataset"] = dataset

        df["query_key"] = df.apply(
            lambda row: make_query_key(row, dataset),
            axis=1,
        )

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def add_full_query_latency(detail, timings):
    detail = detail.copy()

    # Standardize model names
    detail["model"] = detail["model"].map(pretty_model)

    detail = detail.merge(
        timings,
        on=[
            "dataset",
            "model",
            "query_key",
        ],
        how="left",
    )

    detail["retrieval_latency_seconds"] = pd.to_numeric(
        detail["latency_seconds"],
        errors="coerce",
    )

    needs_embedding = detail["method"].isin(
        ["Dense", "Hybrid PageIndex"]
    )

    missing = detail[
        needs_embedding
        & detail["embedding_latency_seconds"].isna()
    ]

    if not missing.empty:

        print("\nWARNING: missing query embedding times:")

        print(
            missing[
                [
                    "dataset",
                    "method",
                    "model",
                    "query_key",
                ]
            ]
            .drop_duplicates()
            .head(20)
            .to_string(index=False)
        )

        print(
            f"\nMissing rows: {len(missing)}"
        )

    # BM25 and PageIndex need no embedding time added
    detail.loc[
        ~needs_embedding,
        "embedding_latency_seconds"
    ] = 0.0

    detail["full_query_latency_seconds"] = (
        detail["retrieval_latency_seconds"]
        + detail["embedding_latency_seconds"]
    )

    return detail


# Main configurations only

def select_main_configs(df):
    top_m = pd.to_numeric(
        df.get("top_m"),
        errors="coerce",
    )

    is_bm25 = df["method"] == "BM25"

    is_dense = df["method"] == "Dense"

    is_pageindex = df["method"] == "PageIndex"

    is_hybrid_main = (
        (df["method"] == "Hybrid PageIndex")
        & (df["variant"].astype(str) == "chunk_rerank")
        & (top_m == 10)
    )

    return df[
        is_bm25
        | is_dense
        | is_pageindex
        | is_hybrid_main
    ].copy()


# Aggregate query efficiency + effectiveness

def summarize_query_efficiency(df):
    group_cols = [
        "dataset",
        "method",
        "model",
        "variant",
        "top_m",
    ]

    rows = []

    for keys, group in df.groupby(
        group_cols,
        dropna=False,
    ):

        latency = pd.to_numeric(
            group["full_query_latency_seconds"],
            errors="coerce",
        ).dropna()

        retrieval_latency = pd.to_numeric(
            group["retrieval_latency_seconds"],
            errors="coerce",
        ).dropna()

        embedding_latency = pd.to_numeric(
            group["embedding_latency_seconds"],
            errors="coerce",
        ).dropna()

        row = dict(zip(group_cols, keys))

        row.update({
            "n_queries": group["query_key"].nunique(),

            # Effectiveness
            "accuracy_at_1": pd.to_numeric(
                group["correct_at_1"],
                errors="coerce",
            ).mean(),

            "mrr_at_5": pd.to_numeric(
                group["reciprocal_rank_at_5"],
                errors="coerce",
            ).mean(),

            # Full online query latency
            "mean_query_latency_seconds": latency.mean(),
            "median_query_latency_seconds": latency.median(),
            "p95_query_latency_seconds": latency.quantile(0.95),

            # Diagnostics
            "mean_embedding_latency_seconds": (
                embedding_latency.mean()
            ),
            "mean_retrieval_stage_latency_seconds": (
                retrieval_latency.mean()
            ),

            # Query-time LLM tokens
            "avg_input_tokens": pd.to_numeric(
                group["input_tokens"],
                errors="coerce",
            ).mean(),

            "avg_output_tokens": pd.to_numeric(
                group["output_tokens"],
                errors="coerce",
            ).mean(),

            "avg_total_tokens": pd.to_numeric(
                group["total_tokens"],
                errors="coerce",
            ).mean(),
        })

        rows.append(row)

    return pd.DataFrame(rows)


# Separate PageIndex top-1 / top-5 token analysis

def summarize_pageindex_tokens(df):
    page = df[
        df["method"] == "PageIndex"
    ].copy()

    rows = []

    for dataset, group in page.groupby("dataset"):

        row = {
            "dataset": dataset,
            "n_queries": group["query_key"].nunique(),
        }

        token_cols = [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "top1_input_tokens",
            "top1_output_tokens",
            "top1_total_tokens",
            "top5_input_tokens",
            "top5_output_tokens",
            "top5_total_tokens",
        ]

        for col in token_cols:

            values = pd.to_numeric(
                group[col],
                errors="coerce",
            )

            row[f"avg_{col}"] = values.mean()
            row[f"total_{col}"] = values.sum()

        rows.append(row)

    return pd.DataFrame(rows)


# Effectiveness-efficiency plots

def method_label(row):
    method = row["method"]
    model = clean_string(row["model"])

    if method == "BM25":
        return "BM25"

    if method == "PageIndex":
        return "PageIndex"

    if method == "Dense":
        return f"Dense {model}"

    if method == "Hybrid PageIndex":
        return f"Hybrid {model}"

    return f"{method} {model}"


def make_tradeoff_plot(summary, dataset, output_path):
    df = summary[
        summary["dataset"] == dataset
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for _, row in df.iterrows():

        x = row["mean_query_latency_seconds"]
        y = row["mrr_at_5"]

        ax.scatter(
            x,
            y,
            s=70,
        )

        ax.annotate(
            method_label(row),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    # Use log scale because of the large latency differences
    ax.set_xscale("log")

    ax.set_xlabel(
        "Mean query latency (seconds, log scale)"
    )

    ax.set_ylabel("MRR@5")

    ax.set_title(
        f"Effectiveness–efficiency trade-off: {dataset}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# Main

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n=== FINDING PREPROCESSING RUNS ===")

    dense_setup, hybrid_setup = find_setup_rows()

    print(
        f"Dense valid setup rows: {len(dense_setup)}"
    )

    print(
        f"Hybrid valid setup rows: {len(hybrid_setup)}"
    )

    # Preprocessing efficiency

    preprocessing = build_preprocessing_table(
        dense_setup,
        hybrid_setup,
    )

    preprocessing.to_csv(
        OUTPUT_DIR / "preprocessing_efficiency_components.csv",
        index=False,
    )

    print("\n=== PREPROCESSING EFFICIENCY ===")
    print(
        preprocessing.to_string(index=False)
    )

    # Query-time efficiency

    print("\n=== LOADING QUERY EMBEDDING TIMES ===")

    timings = discover_query_embedding_timings()

    detail = load_query_results()

    detail = add_full_query_latency(
        detail,
        timings,
    )

    detail.to_csv(
        OUTPUT_DIR / "query_level_full_latency.csv",
        index=False,
    )

    # All configurations
    all_summary = summarize_query_efficiency(detail)

    all_summary.to_csv(
        OUTPUT_DIR / "query_time_efficiency_all.csv",
        index=False,
    )

    # Main thesis comparison:
    # BM25
    # Dense Harrier/Linq
    # PageIndex
    # Hybrid chunk-rerank m=10 Harrier/Linq
    main_detail = select_main_configs(detail)

    main_summary = summarize_query_efficiency(
        main_detail
    )

    main_summary.to_csv(
        OUTPUT_DIR / "query_time_efficiency_main.csv",
        index=False,
    )

    print("\n=== MAIN QUERY-TIME EFFICIENCY ===")

    display_cols = [
        "dataset",
        "method",
        "model",
        "n_queries",
        "accuracy_at_1",
        "mrr_at_5",
        "mean_query_latency_seconds",
        "median_query_latency_seconds",
        "p95_query_latency_seconds",
        "mean_embedding_latency_seconds",
        "mean_retrieval_stage_latency_seconds",
        "avg_input_tokens",
        "avg_output_tokens",
    ]

    print(
        main_summary[
            display_cols
        ].to_string(index=False)
    )

    # PageIndex token usage

    pageindex_tokens = summarize_pageindex_tokens(
        main_detail
    )

    pageindex_tokens.to_csv(
        OUTPUT_DIR / "pageindex_query_token_usage.csv",
        index=False,
    )

    print("\n=== PAGEINDEX QUERY TOKEN USAGE ===")

    print(
        pageindex_tokens.to_string(index=False)
    )

    # Effectiveness-efficiency table

    tradeoff_cols = [
        "dataset",
        "method",
        "model",
        "variant",
        "top_m",
        "accuracy_at_1",
        "mrr_at_5",
        "mean_query_latency_seconds",
        "median_query_latency_seconds",
        "p95_query_latency_seconds",
        "avg_input_tokens",
        "avg_output_tokens",
        "avg_total_tokens",
    ]

    effectiveness_efficiency = (
        main_summary[tradeoff_cols].copy()
    )

    effectiveness_efficiency.to_csv(
        OUTPUT_DIR / "effectiveness_efficiency_main.csv",
        index=False,
    )

    # Plots

    for dataset in ["ESRS", "FinanceBench"]:

        make_tradeoff_plot(
            main_summary,
            dataset,
            OUTPUT_DIR
            / f"{dataset}_mrr_latency_tradeoff.png",
        )

    print(
        f"\nSaved efficiency outputs to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()