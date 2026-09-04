import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


CATEGORIES = [
    "correct",
    "partially_correct",
    "incorrect",
    "no_answer",
]


def read_jsonl(path):
    rows = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {exc}"
                ) from exc

    return rows


def latest_successful_generations(path):
    """Keep the latest successful generation for each answer_id."""
    latest = {}

    for row in read_jsonl(path):
        answer_id = row.get("answer_id")

        if not answer_id:
            continue

        generated_answer = str(row.get("generated_answer") or "").strip()

        if row.get("generation_success") and generated_answer:
            latest[str(answer_id)] = row

    return latest


def latest_successful_judgements(path):
    """
    Keep the latest successful judgement for each answer_id.
    """
    latest = {}

    for row in read_jsonl(path):
        answer_id = row.get("answer_id")

        if not answer_id:
            continue

        if row.get("judge_success"):
            latest[str(answer_id)] = row

    return latest


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to write: {path}")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    """
    Summarize answer quality conditional on whether the
    ground-truth chunk was retrieved.
    """

    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["dataset"],
            row["configuration_id"],
            row["method"],
            row["retrieval_model"],
            row["variant"],
            row["top_m"],
            row["retrieval_setting"],
            row["correct_chunk_retrieved"],
        )

        grouped[key].append(row)

    output = []

    for key, group in sorted(
        grouped.items(),
        key=lambda x: tuple(str(v) for v in x[0]),
    ):
        counts = defaultdict(int)

        for row in group:
            counts[row["judge_category"]] += 1

        n = len(group)

        result = {
            "dataset": key[0],
            "configuration_id": key[1],
            "method": key[2],
            "retrieval_model": key[3],
            "variant": key[4],
            "top_m": key[5],
            "retrieval_setting": key[6],
            "correct_chunk_retrieved": key[7],
            "n": n,
        }

        for category in CATEGORIES:
            count = counts[category]

            result[f"{category}_count"] = count
            result[f"{category}_pct"] = (
                round(100 * count / n, 3) if n else 0
            )

        output.append(result)

    return output


def summarize_retrieval_success(rows):
    """
    Overall retrieval-success rates per configuration and retrieval setting.
    """

    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["dataset"],
            row["configuration_id"],
            row["method"],
            row["retrieval_model"],
            row["variant"],
            row["top_m"],
            row["retrieval_setting"],
        )

        grouped[key].append(row)

    output = []

    for key, group in sorted(
        grouped.items(),
        key=lambda x: tuple(str(v) for v in x[0]),
    ):
        n = len(group)

        retrieved = sum(
            1 for row in group
            if row["correct_chunk_retrieved"] == "yes"
        )

        not_retrieved = n - retrieved

        output.append({
            "dataset": key[0],
            "configuration_id": key[1],
            "method": key[2],
            "retrieval_model": key[3],
            "variant": key[4],
            "top_m": key[5],
            "retrieval_setting": key[6],
            "n": n,
            "correct_chunk_retrieved_count": retrieved,
            "correct_chunk_retrieved_pct": round(100 * retrieved / n, 3),
            "correct_chunk_not_retrieved_count": not_retrieved,
            "correct_chunk_not_retrieved_pct": round(
                100 * not_retrieved / n, 3
            ),
        })

    return output



def summarize_top1_top5_retrieval(rows):
    """Summarize ground-truth retrieval at Top-1 and Top-5."""
    grouped = defaultdict(lambda: {"top_1": [], "top_5": []})

    for row in rows:
        setting = row.get("retrieval_setting")
        if setting not in {"top_1", "top_5"}:
            continue

        key = (
            row["dataset"],
            row["configuration_id"],
            row["method"],
            row["retrieval_model"],
            row["variant"],
            row["top_m"],
        )
        grouped[key][setting].append(row)

    output = []

    for key, settings in sorted(
        grouped.items(),
        key=lambda x: tuple(str(v) for v in x[0]),
    ):
        top1 = settings["top_1"]
        top5 = settings["top_5"]

        n_top1 = len(top1)
        n_top5 = len(top5)

        gt_top1_count = sum(
            1 for row in top1 if row["correct_chunk_retrieved"] == "yes"
        )
        gt_top5_count = sum(
            1 for row in top5 if row["correct_chunk_retrieved"] == "yes"
        )

        if n_top1 != n_top5:
            print(
                "WARNING: different numbers of Top-1 and Top-5 rows for "
                f"{key}: top_1={n_top1}, top_5={n_top5}"
            )

        output.append({
            "dataset": key[0],
            "configuration_id": key[1],
            "method": key[2],
            "retrieval_model": key[3],
            "variant": key[4],
            "top_m": key[5],
            "n_top1": n_top1,
            "gt_retrieved_top1_count": gt_top1_count,
            "gt_retrieved_top1_pct": (
                round(100 * gt_top1_count / n_top1, 3) if n_top1 else 0
            ),
            "n_top5": n_top5,
            "gt_retrieved_top5_count": gt_top5_count,
            "gt_retrieved_top5_pct": (
                round(100 * gt_top5_count / n_top5, 3) if n_top5 else 0
            ),
        })

    return output

def add_combined_rows(rows):
    """
    Pool top-1 and top-5 for descriptive analysis only.
    Combined is not a separate retrieval setting.
    """

    combined = []

    for row in rows:
        copy_row = dict(row)
        copy_row["retrieval_setting"] = "combined"
        combined.append(copy_row)

    return rows + combined


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Join generated end-to-end answers with LLM judgements and "
            "analyse answer quality conditional on retrieval success."
        )
    )

    parser.add_argument(
        "--generated",
        required=True,
        help="Path to generated_answers.jsonl",
    )

    parser.add_argument(
        "--judgements",
        required=True,
        help="Path to judgement JSONL",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for output CSV files",
    )

    parser.add_argument(
        "--include-combined",
        action="store_true",
        help=(
            "Also create descriptive pooled rows combining top_1 and top_5. "
            "Do not interpret this as a separate retrieval setting."
        ),
    )

    args = parser.parse_args()

    generations = latest_successful_generations(args.generated)
    judgements = latest_successful_judgements(args.judgements)

    print(f"Successful generated answers: {len(generations)}")
    print(f"Successful judgements:       {len(judgements)}")

    joined = []

    missing_generation = 0

    for answer_id, judgement in judgements.items():

        generation = generations.get(answer_id)

        if generation is None:
            missing_generation += 1
            continue

        ground_truth_chunk_id = str(
            generation.get("ground_truth_chunk_id") or ""
        ).strip()

        retrieved_chunk_ids = [
            str(x).strip()
            for x in (generation.get("retrieved_chunk_ids") or [])
        ]

        correct_chunk_retrieved = (
            ground_truth_chunk_id != ""
            and ground_truth_chunk_id in retrieved_chunk_ids
        )

        joined.append({
            "answer_id": answer_id,
            "dataset": generation.get("dataset"),
            "configuration_id": generation.get("configuration_id"),
            "method": generation.get("method"),
            "retrieval_model": generation.get("retrieval_model"),
            "variant": generation.get("variant"),
            "top_m": generation.get("top_m"),
            "query_id": generation.get("query_id"),
            "financebench_id": generation.get("financebench_id"),
            "doc_name": generation.get("doc_name"),
            "retrieval_setting": generation.get("retrieval_setting"),
            "top_k": generation.get("top_k"),
            "ground_truth_chunk_id": ground_truth_chunk_id,
            "retrieved_chunk_ids": "|".join(retrieved_chunk_ids),
            "correct_chunk_retrieved": (
                "yes" if correct_chunk_retrieved else "no"
            ),
            "judge_category": judgement.get("judge_category"),
        })

    print(f"Successfully joined:         {len(joined)}")
    print(f"Judgements without generation row: {missing_generation}")

    # Safety check
    invalid_categories = sorted({
        row["judge_category"]
        for row in joined
        if row["judge_category"] not in CATEGORIES
    })

    if invalid_categories:
        print(
            "WARNING: unexpected judge categories:",
            invalid_categories,
        )

    output_dir = Path(args.output_dir)

    # Query-level results

    write_csv(
        output_dir / "end_to_end_retrieval_joined.csv",
        joined,
    )

    # Conditional answer quality by retrieval success

    analysis_rows = joined

    if args.include_combined:
        analysis_rows = add_combined_rows(joined)

    conditional_summary = summarize(analysis_rows)

    write_csv(
        output_dir / "end_to_end_by_retrieval_success.csv",
        conditional_summary,
    )

    # Retrieval success rates

    retrieval_summary = summarize_retrieval_success(
        analysis_rows
    )

    write_csv(
        output_dir / "end_to_end_retrieval_success_rates.csv",
        retrieval_summary,
    )

    # Top-1 and Top-5 retrieval summary

    top1_top5_summary = summarize_top1_top5_retrieval(joined)

    write_csv(
        output_dir / "end_to_end_gt_retrieval_top1_top5.csv",
        top1_top5_summary,
    )

    print("\nCreated:")
    print(
        output_dir / "end_to_end_retrieval_joined.csv"
    )
    print(
        output_dir / "end_to_end_by_retrieval_success.csv"
    )
    print(
        output_dir / "end_to_end_retrieval_success_rates.csv"
    )
    print(
        output_dir / "end_to_end_gt_retrieval_top1_top5.csv"
    )


if __name__ == "__main__":
    main()