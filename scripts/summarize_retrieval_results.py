import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_chunk_ids(ids):
    if not ids:
        return []
    return [str(x) for x in ids]


def first_nonempty(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return ""


def query_text(row):
    return first_nonempty(row.get("query"), row.get("question"))


def target_doc_name(row):
    return str(first_nonempty(row.get("doc_name"), row.get("bank_name")))


def get_query_key(row, dataset):
    """
    FinanceBench: use financebench_id.
    ESRS: use stable content-based key shared across retrieval methods.
    """
    financebench_id = row.get("financebench_id")
    if dataset == "FinanceBench" and financebench_id not in (None, ""):
        return str(financebench_id)

    return (
        f"{target_doc_name(row)}__"
        f"{row.get('ground_truth_chunk_id', '')}__"
        f"{query_text(row)}"
    )

def normalize_doc_key(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    value = str(value).strip()

    # Remove path if present
    value = Path(value).name

    # Remove .pdf extension if present
    if value.lower().endswith(".pdf"):
        value = value[:-4]

    # Normalize common naming differences
    value = value.upper()
    value = value.replace("&", "_")
    value = value.replace(" ", "_")
    value = value.replace("-", "_")

    # Collapse repeated underscores
    while "__" in value:
        value = value.replace("__", "_")

    return value.strip("_")

def normalize_yes_no(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    value = str(value).strip().lower()

    # No TOC
    if value in {"no", "n", "false", "0", "absent", "no toc"}:
        return "No TOC"

    # Any form of TOC presence counts as TOC,
    # regardless of whether page indices are provided
    if (
        value in {"yes", "y", "true", "1", "present", "toc"}
        or "toc" in value
    ):
        return "Present"

    return str(value)


# ---------------------------------------------------------------------
# Result-file discovery
# ---------------------------------------------------------------------

def discover_result_files(results_dir):
    """
    Select final retrieval JSONL files and avoid ambiguous/duplicate artifacts.

    Preferred files:
      - bm25_query_results.jsonl
      - dense_*_query_results.jsonl
      - pageindex_combined_query_results_clean.jsonl if present,
        otherwise pageindex_combined_query_results.jsonl
      - hybrid files with explicit _top_m_X_ in filename

    Generic hybrid files without explicit top_m are only used when no explicit
    top_m runs exist for that same model/variant.
    """
    results_dir = Path(results_dir)
    all_jsonl = sorted(results_dir.glob("*.jsonl"))

    selected = []

    # Baselines
    bm25 = results_dir / "bm25_query_results.jsonl"
    if bm25.exists():
        selected.append(bm25)

    selected.extend(sorted(results_dir.glob("dense_*_query_results.jsonl")))

    # PageIndex: prefer cleaned ESRS file when available
    pageindex_clean = results_dir / "pageindex_combined_query_results_clean.jsonl"
    pageindex_raw = results_dir / "pageindex_combined_query_results.jsonl"
    if pageindex_clean.exists():
        selected.append(pageindex_clean)
    elif pageindex_raw.exists():
        selected.append(pageindex_raw)

    # Hybrid runs with explicit top_m
    explicit_hybrids = [
        p for p in all_jsonl
        if p.name.startswith("hybrid_pageindex")
        and "_top_m_" in p.name
        and p.name.endswith("_query_results.jsonl")
    ]
    selected.extend(explicit_hybrids)

    # If a model/variant has no explicit top_m runs, retain its generic run.
    generic_hybrids = [
        p for p in all_jsonl
        if p.name.startswith("hybrid_pageindex")
        and "_top_m_" not in p.name
        and p.name.endswith("_query_results.jsonl")
    ]

    def hybrid_family(filename):
        name = filename
        name = re.sub(r"_top_m_\d+_query_results\.jsonl$", "", name)
        name = re.sub(r"_query_results\.jsonl$", "", name)
        return name

    explicit_families = {hybrid_family(p.name) for p in explicit_hybrids}
    for p in generic_hybrids:
        if hybrid_family(p.name) not in explicit_families:
            selected.append(p)

    # Preserve order, remove duplicates.
    out = []
    seen = set()
    for p in selected:
        resolved = str(p.resolve())
        if resolved not in seen:
            seen.add(resolved)
            out.append(p)

    return out


# ---------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------

def pretty_model_name(model):
    model = str(model or "")
    low = model.lower()

    if "harrier" in low:
        return "Harrier"
    if "linq" in low and "mistral" in low:
        return "Linq-Embed-Mistral"
    if "gpt-5" in low:
        return "GPT-5"
    if "bm25" in low:
        return "BM25"
    return model or "–"


def pretty_method_name(method, filename):
    low = str(method or "").lower()
    fn = filename.lower()

    if "chunk_rerank" in low or "chunk_rerank" in fn:
        return "Hybrid PageIndex"
    if "hybrid_pageindex" in low or fn.startswith("hybrid_pageindex"):
        return "Hybrid PageIndex"
    if "pageindex" in low or fn.startswith("pageindex"):
        return "PageIndex"
    if "dense" in low or fn.startswith("dense_"):
        return "Dense"
    if "bm25" in low or fn.startswith("bm25"):
        return "BM25"
    return method or filename


def infer_variant(method, mode, filename):
    low_method = str(method or "").lower()
    low_mode = str(mode or "").lower()
    fn = filename.lower()

    if "chunk_rerank" in low_method or "chunk_rerank" in low_mode or "chunk_rerank" in fn:
        return "chunk_rerank"
    if "hybrid_pageindex" in low_method or fn.startswith("hybrid_pageindex"):
        return "node"
    return "standard"


def infer_top_m(row, filename):
    value = row.get("top_m")
    if value not in (None, ""):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    match = re.search(r"_top_m_(\d+)", filename)
    if match:
        return int(match.group(1))
    return np.nan


# ---------------------------------------------------------------------
# Retrieval metrics and document diagnostics
# ---------------------------------------------------------------------

def compute_retrieval_metrics(row):
    gt = str(row.get("ground_truth_chunk_id", ""))

    top1_ids = normalize_chunk_ids(row.get("top1_chunk_ids"))
    top5_ids = normalize_chunk_ids(row.get("top5_chunk_ids"))[:5]

    correct_at_1 = int(bool(top1_ids) and top1_ids[0] == gt)

    rank_top5 = np.nan
    for rank, chunk_id in enumerate(top5_ids, start=1):
        if chunk_id == gt:
            rank_top5 = rank
            break

    hit_at_5 = int(not pd.isna(rank_top5))
    reciprocal_rank_at_5 = 0.0 if pd.isna(rank_top5) else 1.0 / float(rank_top5)

    return {
        "correct_at_1": correct_at_1,
        "hit_at_5": hit_at_5,
        "ground_truth_rank_top5": rank_top5,
        "reciprocal_rank_at_5": reciprocal_rank_at_5,
        "top1_chunk_id": top1_ids[0] if top1_ids else "",
        "top5_chunk_ids": "|".join(top5_ids),

        # Retrieval-return diagnostics
        "n_top1_returned": len(top1_ids),
        "n_top5_returned": len(top5_ids),
        "no_top1_returned": int(len(top1_ids) == 0),
        "no_top5_returned": int(len(top5_ids) == 0),
        "fewer_than_5_returned": int(len(top5_ids) < 5),
        "no_result_at_all": int(len(top1_ids) == 0 and len(top5_ids) == 0),
    }


def retrieved_doc_from_top1_chunks(row):
    chunks = row.get("top1_chunks") or []
    if not chunks:
        return ""

    first = chunks[0] or {}
    return str(first_nonempty(
        first.get("doc_name"),
        first.get("bank_name"),
        first.get("pdf_name"),
    ))


def infer_doc_from_chunk_id(chunk_id, known_docs):
    chunk_id = str(chunk_id or "")
    if not chunk_id:
        return ""

    # Longest prefix first prevents e.g. similarly named documents matching early.
    for doc in sorted((str(d) for d in known_docs if d), key=len, reverse=True):
        if chunk_id == doc or chunk_id.startswith(doc + "_"):
            return doc
    return ""


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------

def load_query_metadata(path, dataset):
    if path is None or not Path(path).exists():
        return pd.DataFrame()

    rows = load_jsonl(path)
    output = []

    for row in rows:
        output.append({
            "query_id": get_query_key(row, dataset),
            "financebench_id": row.get("financebench_id", ""),
            "company_meta": row.get("company", ""),
            "doc_name_meta": target_doc_name(row),
            "bank_name_meta": row.get("bank_name", ""),
            "question_type": row.get("question_type", ""),
            "question_reasoning": row.get("question_reasoning", ""),
            "query_meta": query_text(row),
            "ground_truth_chunk_id_meta": row.get("ground_truth_chunk_id", ""),
        })

    return pd.DataFrame(output).drop_duplicates("query_id")


def load_structure_metadata(path):
    if path is None or not Path(path).exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "pdf_name" not in df.columns:
        return pd.DataFrame()

    keep = ["pdf_name"]
    for col in [
        "toc_present",
        "num_pages",
        "raw_pdf_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "latency_seconds",
    ]:
        if col in df.columns:
            keep.append(col)

    out = df[keep].copy()
    out = out.rename(columns={
        "pdf_name": "structure_doc_name",
        "toc_present": "toc",
        "num_pages": "document_pages",
        "raw_pdf_tokens": "document_raw_pdf_tokens",
        "input_tokens": "structure_input_tokens",
        "output_tokens": "structure_output_tokens",
        "total_tokens": "structure_total_tokens",
        "latency_seconds": "structure_latency_seconds",
    })

    if "toc" in out.columns:
        out["toc"] = out["toc"].map(normalize_yes_no)

    return out.drop_duplicates("structure_doc_name")


def infer_filing_type(doc_name):
    name = str(doc_name or "").upper()

    if "_10K" in name:
        return "10-K"
    if "_10Q" in name:
        return "10-Q"
    if "_8K" in name:
        return "8-K"
    if "EARNINGS" in name:
        return "Earnings Report"
    return "Other"


# ---------------------------------------------------------------------
# Convert each result file to query-level rows
# ---------------------------------------------------------------------

def rows_to_metrics(path, dataset):
    raw_rows = load_jsonl(path)
    filename = Path(path).name

    # De-duplicate within a file:
    # prefer a completed/successful row if one exists for the same query.
    grouped = {}

    for row in raw_rows:
        query_id = get_query_key(row, dataset)
        if not query_id:
            continue

        is_success = (
            row.get("status") not in {"failed", "error"}
            and not row.get("error")
        )

        existing = grouped.get(query_id)
        if existing is None:
            grouped[query_id] = (is_success, row)
        else:
            existing_success, _ = existing
            # successful row beats failed row; otherwise latest row wins
            if is_success or not existing_success:
                grouped[query_id] = (is_success, row)

    output = []

    for query_id, (is_success, row) in grouped.items():
        gt = row.get("ground_truth_chunk_id")
        if gt in (None, ""):
            continue

        metrics = compute_retrieval_metrics(row)

        method_raw = row.get("method", "")
        model_raw = row.get("model", "")
        mode_raw = row.get("mode", "")

        top1_retrieved_doc = retrieved_doc_from_top1_chunks(row)

        output.append({
            "dataset": dataset,
            "query_id": query_id,
            "financebench_id": row.get("financebench_id", ""),
            "company": row.get("company", ""),
            "doc_name": target_doc_name(row),
            "bank_name": row.get("bank_name", ""),
            "query": query_text(row),
            "ground_truth_chunk_id": str(gt),

            "result_file": filename,
            "method_raw": method_raw,
            "model_raw": model_raw,
            "mode_raw": mode_raw,
            "method": pretty_method_name(method_raw, filename),
            "model": pretty_model_name(model_raw),
            "variant": infer_variant(method_raw, mode_raw, filename),
            "top_m": infer_top_m(row, filename),

            **metrics,

            "retrieved_doc_top1_from_result": top1_retrieved_doc,

            "latency_seconds": pd.to_numeric(row.get("latency_seconds"), errors="coerce"),
            "top1_latency_seconds": pd.to_numeric(row.get("top1_latency_seconds"), errors="coerce"),
            "top5_latency_seconds": pd.to_numeric(row.get("top5_latency_seconds"), errors="coerce"),

            "input_tokens": pd.to_numeric(row.get("input_tokens", 0), errors="coerce"),
            "output_tokens": pd.to_numeric(row.get("output_tokens", 0), errors="coerce"),
            "total_tokens": pd.to_numeric(row.get("total_tokens", 0), errors="coerce"),

            "top1_input_tokens": pd.to_numeric(row.get("top1_input_tokens", 0), errors="coerce"),
            "top1_output_tokens": pd.to_numeric(row.get("top1_output_tokens", 0), errors="coerce"),
            "top1_total_tokens": pd.to_numeric(row.get("top1_total_tokens", 0), errors="coerce"),

            "top5_input_tokens": pd.to_numeric(row.get("top5_input_tokens", 0), errors="coerce"),
            "top5_output_tokens": pd.to_numeric(row.get("top5_output_tokens", 0), errors="coerce"),
            "top5_total_tokens": pd.to_numeric(row.get("top5_total_tokens", 0), errors="coerce"),

            "run_success": int(is_success),
            "error": row.get("error", ""),
        })

    return pd.DataFrame(output)


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

CONFIG_COLS = ["dataset", "result_file", "method", "model", "variant", "top_m"]


def add_percentage_columns(df):
    if df.empty:
        return df

    mapping = {
        "accuracy_at_1": "accuracy_at_1_pct",
        "hit_at_5": "hit_at_5_pct",
        "mrr_at_5": "mrr_at_5_pct",
        "document_accuracy_at_1": "document_accuracy_at_1_pct",
    }

    for raw, pct in mapping.items():
        if raw in df.columns:
            df[pct] = df[raw] * 100.0

    return df


def aggregate(df, extra_group_cols=None):
    if extra_group_cols is None:
        extra_group_cols = []

    group_cols = CONFIG_COLS + list(extra_group_cols)

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_queries=("query_id", "nunique"),
            accuracy_at_1=("correct_at_1", "mean"),
            hit_at_5=("hit_at_5", "mean"),
            mrr_at_5=("reciprocal_rank_at_5", "mean"),
            document_accuracy_at_1=("document_correct_at_1", "mean"),

            avg_latency_seconds=("latency_seconds", "mean"),
            avg_top1_latency_seconds=("top1_latency_seconds", "mean"),
            avg_top5_latency_seconds=("top5_latency_seconds", "mean"),

            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),

            avg_top1_input_tokens=("top1_input_tokens", "mean"),
            avg_top1_output_tokens=("top1_output_tokens", "mean"),
            avg_top1_total_tokens=("top1_total_tokens", "mean"),

            avg_top5_input_tokens=("top5_input_tokens", "mean"),
            avg_top5_output_tokens=("top5_output_tokens", "mean"),
            avg_top5_total_tokens=("top5_total_tokens", "mean"),

            successful_rows=("run_success", "sum"),
        )
        .reset_index()
    )

    return add_percentage_columns(out)


def latency_statistics(df):
    rows = []

    for keys, group in df.groupby(CONFIG_COLS, dropna=False):
        values = pd.to_numeric(group["latency_seconds"], errors="coerce").dropna()

        row = dict(zip(CONFIG_COLS, keys))
        row.update({
            "n_queries": group["query_id"].nunique(),
            "n_latency_observations": len(values),
            "mean_latency_seconds": values.mean() if len(values) else np.nan,
            "median_latency_seconds": values.median() if len(values) else np.nan,
            "std_latency_seconds": values.std(ddof=1) if len(values) > 1 else 0.0,
            "p95_latency_seconds": values.quantile(0.95) if len(values) else np.nan,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# PageIndex disagreement diagnostics
# ---------------------------------------------------------------------

def pageindex_diagnostic_category(row):
    acc = int(row["correct_at_1"]) == 1
    rank = row["ground_truth_rank_top5"]

    if acc and rank == 1:
        return "both_rank1"
    if (not acc) and rank == 1:
        return "mrr_rank1_only"
    if acc and pd.notna(rank) and 2 <= rank <= 5:
        return "accuracy_only_mrr_lower"
    if acc and pd.isna(rank):
        return "accuracy_only_mrr_missing"
    if (not acc) and pd.notna(rank) and 2 <= rank <= 5:
        return "mrr_lower_only"
    return "both_fail"


def make_pageindex_diagnostics(df):
    page = df[df["method"] == "PageIndex"].copy()

    if page.empty:
        return pd.DataFrame(), pd.DataFrame()

    page["diagnostic_category"] = page.apply(pageindex_diagnostic_category, axis=1)

    detail_cols = [
        "dataset",
        "query_id",
        "financebench_id",
        "company",
        "doc_name",
        "bank_name",
        "query",
        "ground_truth_chunk_id",
        "top1_chunk_id",
        "top5_chunk_ids",
        "correct_at_1",
        "ground_truth_rank_top5",
        "diagnostic_category",
        "latency_seconds",
        "top1_latency_seconds",
        "top5_latency_seconds",
        "top1_input_tokens",
        "top1_output_tokens",
        "top5_input_tokens",
        "top5_output_tokens",
    ]
    detail_cols = [c for c in detail_cols if c in page.columns]
    detail = page[detail_cols].copy()

    summary = (
        page.groupby(CONFIG_COLS + ["diagnostic_category"], dropna=False)
        .agg(count=("query_id", "nunique"))
        .reset_index()
    )

    totals = (
        summary.groupby(CONFIG_COLS, dropna=False)["count"]
        .transform("sum")
    )
    summary["percentage"] = summary["count"] / totals * 100.0

    return summary, detail


def make_pageindex_return_diagnostics(df):
    page = df[df["method"] == "PageIndex"].copy()

    if page.empty:
        return pd.DataFrame()

    summary = (
        page.groupby(CONFIG_COLS, dropna=False)
        .agg(
            n_queries=("query_id", "nunique"),
            no_top1_count=("no_top1_returned", "sum"),
            no_top5_count=("no_top5_returned", "sum"),
            fewer_than_5_count=("fewer_than_5_returned", "sum"),
            no_result_at_all_count=("no_result_at_all", "sum"),
            mean_top5_results=("n_top5_returned", "mean"),
        )
        .reset_index()
    )

    summary["no_top1_pct"] = (
        summary["no_top1_count"] / summary["n_queries"] * 100
    )

    summary["no_top5_pct"] = (
        summary["no_top5_count"] / summary["n_queries"] * 100
    )

    summary["fewer_than_5_pct"] = (
        summary["fewer_than_5_count"] / summary["n_queries"] * 100
    )

    summary["no_result_at_all_pct"] = (
        summary["no_result_at_all_count"] / summary["n_queries"] * 100
    )

    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create thesis-ready retrieval-result tables for ESRS or FinanceBench."
    )
    parser.add_argument("--dataset", required=True, choices=["ESRS", "FinanceBench"])
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--structure-runs", default=None)
    args = parser.parse_args()

    dataset = args.dataset
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default metadata locations when run from repo root.
    if args.queries is None:
        if dataset == "ESRS":
            queries_path = Path("final results/ESRS queries/ESRS_queries.jsonl")
        else:
            queries_path = Path("final results/FinanceBench queries/FinanceBench_queries.jsonl")
    else:
        queries_path = Path(args.queries)

    if args.structure_runs is None:
        structure_path = Path(f"final results/{dataset}_pageindex_structure_runs.csv")
    else:
        structure_path = Path(args.structure_runs)

    files = discover_result_files(results_dir)
    if not files:
        raise FileNotFoundError(f"No retrieval result JSONL files found in {results_dir}")

    print("\nSelected result files:")
    for p in files:
        print(f"  - {p.name}")

    frames = []
    for p in files:
        df = rows_to_metrics(p, dataset)
        print(f"{p.name}: {len(df)} query rows")
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No valid retrieval rows were found.")

    detail = pd.concat(frames, ignore_index=True)

    # Join master query metadata.
    query_meta = load_query_metadata(queries_path, dataset)
    if not query_meta.empty:
        detail = detail.merge(query_meta, on="query_id", how="left")

        # Fill missing row metadata from master metadata.
        for target, source in [
            ("financebench_id", "financebench_id_y"),
            ("company", "company_meta"),
            ("doc_name", "doc_name_meta"),
            ("bank_name", "bank_name_meta"),
            ("query", "query_meta"),
            ("ground_truth_chunk_id", "ground_truth_chunk_id_meta"),
        ]:
            if source in detail.columns:
                if target in detail.columns:
                    detail[target] = detail[target].replace("", np.nan).fillna(detail[source])
                else:
                    detail[target] = detail[source]

        # pandas may suffix financebench_id during merge
        if "financebench_id_x" in detail.columns:
            detail["financebench_id"] = (
                detail["financebench_id_x"]
                .replace("", np.nan)
                .fillna(detail.get("financebench_id_y"))
            )

        drop_cols = [
            "financebench_id_x",
            "financebench_id_y",
            "company_meta",
            "doc_name_meta",
            "bank_name_meta",
            "query_meta",
            "ground_truth_chunk_id_meta",
        ]
        detail = detail.drop(columns=[c for c in drop_cols if c in detail.columns])

    # Join structure metadata by normalized target document name
    structure_meta = load_structure_metadata(structure_path)

    if not structure_meta.empty:

        detail["doc_merge_key"] = detail["doc_name"].map(normalize_doc_key)

        structure_meta["doc_merge_key"] = (
            structure_meta["structure_doc_name"]
            .map(normalize_doc_key)
        )

        detail = detail.merge(
            structure_meta,
            on="doc_merge_key",
            how="left",
        )

        detail = detail.drop(
            columns=["structure_doc_name", "doc_merge_key"],
            errors="ignore"
        )

    # FinanceBench-specific metadata.
    if dataset == "FinanceBench":
        detail["filing_type"] = detail["doc_name"].map(infer_filing_type)

    # Correct-document diagnostic.
    known_docs = set(detail["doc_name"].dropna().astype(str).unique())
    detail["retrieved_doc_top1"] = detail["retrieved_doc_top1_from_result"]

    missing_doc = detail["retrieved_doc_top1"].isin(["", None]) | detail["retrieved_doc_top1"].isna()
    detail.loc[missing_doc, "retrieved_doc_top1"] = detail.loc[missing_doc, "top1_chunk_id"].map(
        lambda x: infer_doc_from_chunk_id(x, known_docs)
    )

    detail["document_correct_at_1"] = (
        detail["retrieved_doc_top1"].astype(str) == detail["doc_name"].astype(str)
    ).astype(int)

    # Coverage audit against master query file.
    expected_n = query_meta["query_id"].nunique() if not query_meta.empty else np.nan

    coverage = (
        detail.groupby(CONFIG_COLS, dropna=False)
        .agg(n_result_queries=("query_id", "nunique"))
        .reset_index()
    )
    coverage["master_n_queries"] = expected_n
    if pd.notna(expected_n) and expected_n > 0:
        coverage["coverage_pct"] = coverage["n_result_queries"] / expected_n * 100.0
    else:
        coverage["coverage_pct"] = np.nan

    # Main outputs.
    overall = aggregate(detail)
    overall = overall.merge(
        coverage,
        on=CONFIG_COLS,
        how="left",
        suffixes=("", "_coverage"),
    )

    latency = latency_statistics(detail)

    # ---------------------------------------------------------
    # Retrieval by TOC availability - both datasets
    # ---------------------------------------------------------

    if "toc" in detail.columns:
        by_toc = aggregate(detail, ["toc"])
        by_toc.to_csv(output_dir / "retrieval_by_toc.csv", index=False)


    # ---------------------------------------------------------
    # Dataset-specific outputs
    # ---------------------------------------------------------

    if dataset == "ESRS":

        by_bank = aggregate(detail, ["doc_name"])
        by_bank = by_bank.rename(columns={"doc_name": "bank"})
        by_bank.to_csv(output_dir / "retrieval_by_bank.csv", index=False)

    else:

        by_filing = aggregate(detail, ["filing_type"])
        by_filing.to_csv(output_dir / "retrieval_by_filing_type.csv", index=False)

        if "question_type" in detail.columns:
            by_qtype = aggregate(detail, ["question_type"])
            by_qtype.to_csv(output_dir / "retrieval_by_question_type.csv", index=False)

        if "question_reasoning" in detail.columns:
            by_reasoning = aggregate(detail, ["question_reasoning"])
            by_reasoning.to_csv(
                output_dir / "retrieval_by_question_reasoning.csv",
                index=False,
            )

        by_doc = aggregate(detail, ["doc_name"])
        by_doc.to_csv(output_dir / "retrieval_by_document.csv", index=False)

        if "question_type" in detail.columns:
            by_qtype = aggregate(detail, ["question_type"])
            by_qtype.to_csv(output_dir / "retrieval_by_question_type.csv", index=False)

        if "question_reasoning" in detail.columns:
            by_reasoning = aggregate(detail, ["question_reasoning"])
            by_reasoning.to_csv(
                output_dir / "retrieval_by_question_reasoning.csv",
                index=False,
            )

        by_doc = aggregate(detail, ["doc_name"])
        by_doc.to_csv(output_dir / "retrieval_by_document.csv", index=False)

    # Hybrid top-m sensitivity.
    hybrid = overall[
        (overall["method"] == "Hybrid PageIndex")
        & overall["top_m"].notna()
    ].copy()
    hybrid.to_csv(output_dir / "hybrid_top_m_sensitivity.csv", index=False)

    # PageIndex disagreement analysis.
    diag_summary, diag_detail = make_pageindex_diagnostics(detail)
    return_diag = make_pageindex_return_diagnostics(detail)

    return_diag.to_csv(
        output_dir / "pageindex_return_diagnostics.csv",
        index=False,
    )
    diag_summary.to_csv(
        output_dir / "pageindex_accuracy_mrr_diagnostics.csv",
        index=False,
    )
    diag_detail.to_csv(
        output_dir / "pageindex_accuracy_mrr_diagnostic_queries.csv",
        index=False,
    )

    # Save common outputs.
    detail.to_csv(output_dir / "query_level_combined.csv", index=False)
    overall.to_csv(output_dir / "overall_retrieval_summary.csv", index=False)
    latency.to_csv(output_dir / "latency_statistics.csv", index=False)
    coverage.to_csv(output_dir / "result_coverage_audit.csv", index=False)

    print("\n=== OVERALL RETRIEVAL SUMMARY ===")
    display_cols = [
        "method",
        "model",
        "variant",
        "top_m",
        "n_queries",
        "accuracy_at_1_pct",
        "hit_at_5_pct",
        "mrr_at_5_pct",
        "document_accuracy_at_1_pct",
        "avg_latency_seconds",
        "avg_input_tokens",
        "avg_output_tokens",
        "coverage_pct",
    ]
    display_cols = [c for c in display_cols if c in overall.columns]
    print(overall[display_cols].to_string(index=False))

    print(f"\nSaved analysis files to: {output_dir}")


if __name__ == "__main__":
    main()