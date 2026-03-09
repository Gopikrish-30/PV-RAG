"""
PV-RAG Experiment Analysis & Visualization

Generates publication-quality charts and statistical analysis:
  1. Radar chart: All metrics comparison across systems
  2. Bar charts: Per-category and per-difficulty breakdown
  3. Heatmap: System × Metric performance matrix
  4. Temporal analysis: Performance across different time periods
  5. Ablation study: Component contribution analysis
  6. Confidence calibration: Reliability diagram
  7. Latency comparison: Response time analysis
  8. Statistical tests: Paired t-test / Wilcoxon for significance

Usage:
  python -m experiments.analyze_results
  python -m experiments.analyze_results --results_file experiments/results/experiment_results_XXX.csv
"""
import os
import sys
import glob
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


class ExperimentAnalyzer:
    """Analyze experiment results and generate visualizations."""

    def __init__(self, results_file: Optional[str] = None):
        os.makedirs(PLOTS_DIR, exist_ok=True)

        if results_file:
            self.df = pd.read_csv(results_file)
        else:
            # Load most recent results
            pattern = os.path.join(RESULTS_DIR, "experiment_results_*.csv")
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No results found in {RESULTS_DIR}")
            self.df = pd.read_csv(files[-1])
            logger.info(f"Loaded: {files[-1]}")

        self.systems = sorted(self.df["system"].unique())
        logger.info(f"Systems: {self.systems}")
        logger.info(f"Queries: {self.df['query_id'].nunique()}")

    def run_all_analysis(self):
        """Run all analysis and generate all outputs."""
        logger.info("Running full analysis pipeline...")

        self.generate_overall_comparison_table()
        self.generate_category_analysis()
        self.generate_difficulty_analysis()
        self.generate_ablation_study()
        self.generate_statistical_tests()
        self.generate_temporal_analysis()
        self.generate_novelty_metrics_analysis()

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self.plot_radar_chart()
            self.plot_metric_bars()
            self.plot_heatmap()
            self.plot_category_breakdown()
            self.plot_latency_comparison()
            self.plot_temporal_vs_semantic()
            logger.info(f"All plots saved to: {PLOTS_DIR}")
        except ImportError:
            logger.warning("matplotlib not available — skipping plots (pip install matplotlib)")

        logger.info("Analysis complete!")

    # ──────────────────────────────────────────────────────────────────
    # TEXT-BASED ANALYSIS
    # ──────────────────────────────────────────────────────────────────

    def generate_overall_comparison_table(self):
        """Generate the main comparison table."""
        metrics = [
            "precision_at_1", "precision_at_5", "recall_at_5", "mrr", "ndcg_at_5",
            "temporal_precision_at_1", "temporal_precision_at_5",
            "version_discrimination", "amendment_awareness",
            "temporal_hallucination_rate",
            "keyword_hit_rate", "answer_completeness",
            "latency_seconds",
        ]

        table_data = {}
        for sys_name in self.systems:
            sdf = self.df[self.df["system"] == sys_name]
            row = {}
            for m in metrics:
                if m in sdf.columns:
                    row[m] = sdf[m].dropna().mean()
            table_data[sys_name] = row

        table_df = pd.DataFrame(table_data).T
        table_df = table_df.round(3)

        path = os.path.join(RESULTS_DIR, "overall_comparison.csv")
        table_df.to_csv(path)
        logger.info(f"Overall comparison table: {path}")

        print("\n" + "=" * 80)
        print("OVERALL COMPARISON TABLE")
        print("=" * 80)
        print(table_df.to_string())
        return table_df

    def generate_category_analysis(self):
        """Per-category breakdown of metrics."""
        key_metrics = ["precision_at_1", "temporal_precision_at_1", "keyword_hit_rate"]

        results = []
        for cat in sorted(self.df["category"].unique()):
            for sys_name in self.systems:
                cdf = self.df[(self.df["category"] == cat) & (self.df["system"] == sys_name)]
                row = {"category": cat, "system": sys_name, "n": len(cdf)}
                for m in key_metrics:
                    if m in cdf.columns:
                        row[m] = cdf[m].dropna().mean()
                results.append(row)

        cat_df = pd.DataFrame(results)
        path = os.path.join(RESULTS_DIR, "category_analysis.csv")
        cat_df.to_csv(path, index=False)
        logger.info(f"Category analysis: {path}")

        print("\n" + "=" * 80)
        print("CATEGORY ANALYSIS")
        print("=" * 80)
        for cat in sorted(self.df["category"].unique()):
            cdf_sub = cat_df[cat_df["category"] == cat]
            print(f"\n  {cat}:")
            print(cdf_sub.to_string(index=False))
        return cat_df

    def generate_difficulty_analysis(self):
        """Performance breakdown by query difficulty."""
        key_metrics = ["precision_at_1", "temporal_precision_at_1", "keyword_hit_rate", "temporal_hallucination_rate"]

        results = []
        for diff in ["easy", "medium", "hard"]:
            for sys_name in self.systems:
                ddf = self.df[(self.df["difficulty"] == diff) & (self.df["system"] == sys_name)]
                if ddf.empty:
                    continue
                row = {"difficulty": diff, "system": sys_name, "n": len(ddf)}
                for m in key_metrics:
                    if m in ddf.columns:
                        row[m] = ddf[m].dropna().mean()
                results.append(row)

        diff_df = pd.DataFrame(results)
        path = os.path.join(RESULTS_DIR, "difficulty_analysis.csv")
        diff_df.to_csv(path, index=False)
        logger.info(f"Difficulty analysis: {path}")
        return diff_df

    def generate_ablation_study(self):
        """Component contribution analysis — ablation study."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY: Component Contributions")
        print("=" * 80)

        ablation_systems = ["TemporalOnly", "NaiveRAG", "PV-RAG-NoGraph", "PV-RAG"]
        present_systems = [s for s in ablation_systems if s in self.systems]

        if len(present_systems) < 2:
            print("  Not enough ablation systems present for analysis.")
            return

        ablation_metrics = [
            "temporal_precision_at_1", "version_discrimination",
            "precision_at_1", "keyword_hit_rate"
        ]

        print(f"\n  {'System':<20}", end="")
        for m in ablation_metrics:
            print(f" {m:>22}", end="")
        print()
        print("  " + "-" * 110)

        for sys_name in present_systems:
            sdf = self.df[self.df["system"] == sys_name]
            print(f"  {sys_name:<20}", end="")
            for m in ablation_metrics:
                if m in sdf.columns:
                    val = sdf[m].dropna().mean()
                    print(f" {val:>22.3f}", end="")
            print()

        # Component deltas
        print("\n  Component Contributions (delta from removing component):")
        pvrag = self.df[self.df["system"] == "PV-RAG"]
        if pvrag.empty:
            return

        for sys_name in present_systems:
            if sys_name == "PV-RAG":
                continue
            other = self.df[self.df["system"] == sys_name]
            if other.empty:
                continue

            component = {
                "NaiveRAG": "Temporal Filtering + Version Chains + Graph",
                "TemporalOnly": "Semantic Ranking + Hybrid Merge",
                "PV-RAG-NoGraph": "ILO-Graph Augmentation",
            }.get(sys_name, sys_name)

            tp1_pv = pvrag["temporal_precision_at_1"].dropna().mean()
            tp1_other = other["temporal_precision_at_1"].dropna().mean()
            delta = tp1_pv - tp1_other

            print(f"  • Removing {component}: TP@1 drops by {delta:.3f}")

    def generate_statistical_tests(self):
        """Paired statistical tests: PV-RAG vs each baseline."""
        from scipy import stats

        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 80)

        pvrag = self.df[self.df["system"] == "PV-RAG"]
        if pvrag.empty:
            print("  PV-RAG results not found.")
            return

        test_metrics = ["precision_at_1", "temporal_precision_at_1", "keyword_hit_rate"]

        for sys_name in self.systems:
            if sys_name == "PV-RAG":
                continue

            other = self.df[self.df["system"] == sys_name]
            print(f"\n  PV-RAG vs {sys_name}:")

            for metric in test_metrics:
                if metric not in pvrag.columns or metric not in other.columns:
                    continue

                # Align by query_id
                merged = pvrag[["query_id", metric]].merge(
                    other[["query_id", metric]],
                    on="query_id",
                    suffixes=("_pvrag", "_other"),
                )

                if len(merged) < 3:
                    print(f"    {metric}: insufficient data")
                    continue

                pvrag_vals = merged[f"{metric}_pvrag"].dropna().values
                other_vals = merged[f"{metric}_other"].dropna().values

                if len(pvrag_vals) != len(other_vals):
                    continue

                # Wilcoxon signed-rank test (non-parametric)
                try:
                    stat, p_value = stats.wilcoxon(pvrag_vals, other_vals, alternative="greater")
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    delta = np.mean(pvrag_vals) - np.mean(other_vals)
                    print(
                        f"    {metric:>25}: Δ={delta:+.3f} "
                        f"p={p_value:.4f} {sig}"
                    )
                except Exception:
                    delta = np.mean(pvrag_vals) - np.mean(other_vals)
                    print(f"    {metric:>25}: Δ={delta:+.3f} (test not applicable)")

    def generate_temporal_analysis(self):
        """Analyze performance across different historical periods."""
        print("\n" + "=" * 80)
        print("TEMPORAL ANALYSIS: Performance Across Time Periods")
        print("=" * 80)

        temporal_df = self.df[self.df["expected_year"].notna()].copy()
        if temporal_df.empty:
            print("  No temporal queries in results.")
            return

        temporal_df["era"] = temporal_df["expected_year"].apply(
            lambda y: "Pre-2000" if y < 2000
            else "2000-2010" if y < 2010
            else "2010-2019" if y < 2019
            else "2019-2023" if y < 2024
            else "Post-2024"
        )

        for era in ["Pre-2000", "2000-2010", "2010-2019", "2019-2023", "Post-2024"]:
            era_df = temporal_df[temporal_df["era"] == era]
            if era_df.empty:
                continue
            print(f"\n  Era: {era} (n={era_df['query_id'].nunique()} queries)")
            for sys_name in self.systems:
                sdf = era_df[era_df["system"] == sys_name]
                if sdf.empty:
                    continue
                tp1 = sdf["temporal_precision_at_1"].dropna().mean()
                khr = sdf["keyword_hit_rate"].dropna().mean()
                print(f"    {sys_name:<20}: TP@1={tp1:.3f}  KHR={khr:.3f}")

    def generate_novelty_metrics_analysis(self):
        """Deep analysis of the novel metrics proposed by PV-RAG."""
        print("\n" + "=" * 80)
        print("NOVEL METRICS ANALYSIS (PV-RAG Contributions)")
        print("=" * 80)

        novel_metrics = {
            "temporal_precision_at_1": "Temporal Precision@1 — Correct version for queried year",
            "temporal_precision_at_5": "Temporal Precision@5 — Top-5 temporal validity",
            "version_discrimination": "Version Discrimination Score — Right version, right era",
            "amendment_awareness": "Amendment Awareness — Surfaces amendment history",
            "temporal_hallucination_rate": "Temporal Hallucination Rate — Incorrect temporal claims (lower=better)",
            "cross_era_accuracy": "Cross-Era Accuracy — Correct law for law-replacement boundaries",
        }

        for metric, description in novel_metrics.items():
            if metric not in self.df.columns:
                continue
            print(f"\n  {description}:")
            for sys_name in self.systems:
                sdf = self.df[self.df["system"] == sys_name]
                val = sdf[metric].dropna().mean()
                std = sdf[metric].dropna().std()
                print(f"    {sys_name:<20}: {val:.3f} ± {std:.3f}")

    # ──────────────────────────────────────────────────────────────────
    # VISUALIZATIONS (matplotlib)
    # ──────────────────────────────────────────────────────────────────

    def plot_radar_chart(self):
        """Radar/spider chart comparing all systems on key metrics."""
        import matplotlib.pyplot as plt

        metrics = [
            "precision_at_1", "temporal_precision_at_1", "version_discrimination",
            "amendment_awareness", "keyword_hit_rate", "answer_completeness",
        ]
        labels = ["P@1", "TP@1", "VDS", "AAS", "KHR", "AC"]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        colors = plt.cm.Set2(np.linspace(0, 1, len(self.systems)))

        for i, sys_name in enumerate(self.systems):
            sdf = self.df[self.df["system"] == sys_name]
            values = []
            for m in metrics:
                val = sdf[m].dropna().mean() if m in sdf.columns else 0
                values.append(val)
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=sys_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title("System Comparison — Radar Chart", fontsize=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "radar_chart.png"), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: radar_chart.png")

    def plot_metric_bars(self):
        """Grouped bar chart for key metrics."""
        import matplotlib.pyplot as plt

        metrics = [
            "precision_at_1", "temporal_precision_at_1",
            "keyword_hit_rate", "version_discrimination",
        ]
        labels = ["P@1", "TP@1", "KHR", "VDS"]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(labels))
        width = 0.8 / len(self.systems)
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.systems)))

        for i, sys_name in enumerate(self.systems):
            sdf = self.df[self.df["system"] == sys_name]
            values = [sdf[m].dropna().mean() if m in sdf.columns else 0 for m in metrics]
            offset = (i - len(self.systems) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=sys_name, color=colors[i])

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("Key Metrics — System Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.15)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "metric_bars.png"), dpi=150)
        plt.close()
        logger.info("Saved: metric_bars.png")

    def plot_heatmap(self):
        """System × Metric heatmap."""
        import matplotlib.pyplot as plt

        metrics = [
            "precision_at_1", "precision_at_5", "recall_at_5", "mrr",
            "temporal_precision_at_1", "temporal_precision_at_5",
            "version_discrimination", "amendment_awareness",
            "keyword_hit_rate", "answer_completeness",
        ]
        short_names = ["P@1", "P@5", "R@5", "MRR", "TP@1", "TP@5", "VDS", "AAS", "KHR", "AC"]

        data = []
        for sys_name in self.systems:
            sdf = self.df[self.df["system"] == sys_name]
            row = [sdf[m].dropna().mean() if m in sdf.columns else 0 for m in metrics]
            data.append(row)

        data_arr = np.array(data)

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(data_arr, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(short_names)))
        ax.set_xticklabels(short_names, fontsize=10)
        ax.set_yticks(range(len(self.systems)))
        ax.set_yticklabels(self.systems, fontsize=10)

        for i in range(len(self.systems)):
            for j in range(len(metrics)):
                ax.text(j, i, f"{data_arr[i, j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if data_arr[i, j] > 0.7 else "black")

        ax.set_title("Performance Heatmap: System × Metric", fontsize=13)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Score")

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "heatmap.png"), dpi=150)
        plt.close()
        logger.info("Saved: heatmap.png")

    def plot_category_breakdown(self):
        """Bar chart: Performance by query category for PV-RAG vs NaiveRAG."""
        import matplotlib.pyplot as plt

        compare_systems = [s for s in ["PV-RAG", "NaiveRAG", "LLMOnly"] if s in self.systems]
        if not compare_systems:
            return

        categories = sorted(self.df["category"].unique())
        metric = "temporal_precision_at_1"

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.8 / len(compare_systems)
        colors = ["#2ecc71", "#e74c3c", "#3498db"]

        for i, sys_name in enumerate(compare_systems):
            values = []
            for cat in categories:
                cdf = self.df[(self.df["category"] == cat) & (self.df["system"] == sys_name)]
                values.append(cdf[metric].dropna().mean() if not cdf.empty and metric in cdf.columns else 0)

            offset = (i - len(compare_systems) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=sys_name, color=colors[i % len(colors)])

        ax.set_xlabel("Query Category")
        ax.set_ylabel("Temporal Precision@1")
        ax.set_title("Temporal Precision by Query Category")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "category_breakdown.png"), dpi=150)
        plt.close()
        logger.info("Saved: category_breakdown.png")

    def plot_latency_comparison(self):
        """Box plot of response latencies by system."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        data = []
        labels = []
        for sys_name in self.systems:
            sdf = self.df[self.df["system"] == sys_name]
            if "total_time" in sdf.columns:
                data.append(sdf["total_time"].dropna().values)
                labels.append(sys_name)

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)

            ax.set_ylabel("Total Time (seconds)")
            ax.set_title("Response Latency Comparison")
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "latency_comparison.png"), dpi=150)
        plt.close()
        logger.info("Saved: latency_comparison.png")

    def plot_temporal_vs_semantic(self):
        """Scatter: Temporal Precision vs Semantic Precision per system."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        colors = plt.cm.Set2(np.linspace(0, 1, len(self.systems)))
        markers = ["o", "s", "^", "D", "v", "P"]

        for i, sys_name in enumerate(self.systems):
            sdf = self.df[self.df["system"] == sys_name]
            if "precision_at_1" in sdf.columns and "temporal_precision_at_1" in sdf.columns:
                x_vals = sdf["precision_at_1"].dropna().values
                y_vals = sdf["temporal_precision_at_1"].dropna().values
                min_len = min(len(x_vals), len(y_vals))
                ax.scatter(
                    x_vals[:min_len], y_vals[:min_len],
                    label=sys_name, color=colors[i],
                    marker=markers[i % len(markers)], s=60, alpha=0.7,
                )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("Semantic Precision@1")
        ax.set_ylabel("Temporal Precision@1")
        ax.set_title("Semantic vs Temporal Precision")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "temporal_vs_semantic.png"), dpi=150)
        plt.close()
        logger.info("Saved: temporal_vs_semantic.png")


# ──────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze PV-RAG experiment results")
    parser.add_argument("--results_file", help="Path to results CSV file")
    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(results_file=args.results_file)
    analyzer.run_all_analysis()


if __name__ == "__main__":
    main()
