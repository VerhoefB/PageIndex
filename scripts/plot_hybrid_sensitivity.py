from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

esrs_csv = Path(r"final results\ESRS results analysis\overall_retrieval_summary.csv")
financebench_csv = Path(r"final results\FinanceBench results analysis\overall_retrieval_summary.csv")

output_dir = Path(r"results figures")
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load and combine datasets
# ---------------------------------------------------------

esrs = pd.read_csv(esrs_csv)
financebench = pd.read_csv(financebench_csv)

df = pd.concat([esrs, financebench], ignore_index=True)

# Keep only Hybrid PageIndex
hybrid = df[df["method"] == "Hybrid PageIndex"].copy()

hybrid["model_label"] = hybrid["model"].replace({
    "Harrier": "Harrier",
    "Linq-Embed-Mistral": "Linq"
})

hybrid["variant_label"] = hybrid["variant"].replace({
    "node": "Node",
    "chunk_rerank": "Chunk rerank"
})

hybrid["configuration"] = (
    hybrid["model_label"] + " - " + hybrid["variant_label"]
)

hybrid["top_m"] = pd.to_numeric(hybrid["top_m"])


# ---------------------------------------------------------
# Plotting function
# ---------------------------------------------------------

def make_plot(dataset, metric, ylabel, filename):

    subset = hybrid[hybrid["dataset"] == dataset].copy()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    configurations = [
        "Harrier - Node",
        "Harrier - Chunk rerank",
        "Linq - Node",
        "Linq - Chunk rerank",
    ]

    for configuration in configurations:

        temp = subset[
            subset["configuration"] == configuration
        ].sort_values("top_m")

        ax.plot(
            temp["top_m"],
            temp[metric],
            marker="o",
            linewidth=1.8,
            label=configuration
        )

    ax.set_xlabel("$m$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([5, 7, 10])

    if metric == "accuracy_at_1":
        ax.set_ylim(0.15, 0.45)

    elif metric == "mrr_at_5":
        ax.set_ylim(0.24, 0.52)

    ax.grid(alpha=0.25)

    ax.legend(
        frameon=False,
        fontsize=9
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / filename,
        bbox_inches="tight"
    )

    plt.close(fig)


# ---------------------------------------------------------
# Accuracy@1
# ---------------------------------------------------------

make_plot(
    "ESRS",
    "accuracy_at_1",
    "Accuracy@1",
    "hybrid_m_accuracy_esrs.pdf"
)

make_plot(
    "FinanceBench",
    "accuracy_at_1",
    "Accuracy@1",
    "hybrid_m_accuracy_financebench.pdf"
)


# ---------------------------------------------------------
# MRR@5
# ---------------------------------------------------------

make_plot(
    "ESRS",
    "mrr_at_5",
    "MRR@5",
    "hybrid_m_mrr_esrs.pdf"
)

make_plot(
    "FinanceBench",
    "mrr_at_5",
    "MRR@5",
    "hybrid_m_mrr_financebench.pdf"
)

print("Figures saved to:", output_dir)