"""
PV-RAG Experiment Runner — End-to-End Comparative Evaluation

Orchestrates:
  1. Load evaluation dataset (gold-standard QA pairs)
  2. Run each system on every query
  3. Compute all metrics per query
  4. Aggregate results per system, per category, per difficulty
  5. Export results to CSV + JSON
  6. Generate summary tables + visualizations

Usage:
  python -m experiments.run_experiments                    # Full run
  python -m experiments.run_experiments --systems PV-RAG NaiveRAG  # Selective
  python -m experiments.run_experiments --categories temporal_precision  # Filter
  python -m experiments.run_experiments --quick            # 5 queries only
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime, date
from typing import Dict, List, Optional
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.evaluation_dataset import EVALUATION_DATASET
from experiments.metrics import compute_all_metrics, expected_calibration_error
from experiments.baselines import get_system, ALL_SYSTEMS


# ──────────────────────────────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RATE_LIMIT_DELAY = 3.0  # seconds between LLM calls to respect Groq rate limits


# ──────────────────────────────────────────────────────────────────────
# CORE EXPERIMENT RUNNER
# ──────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """Run all systems on the evaluation dataset and collect metrics."""

    def __init__(
        self,
        system_names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        quick: bool = False,
    ):
        self.system_names = system_names or list(ALL_SYSTEMS.keys())
        self.categories = categories
        self.difficulties = difficulties
        self.quick = quick

        # Filter dataset
        self.dataset = self._filter_dataset()

        # Results storage
        self.raw_results: List[Dict] = []  # Per-query results
        self.system_instances: Dict = {}

        os.makedirs(RESULTS_DIR, exist_ok=True)

        logger.info(
            f"ExperimentRunner initialized: "
            f"{len(self.system_names)} systems × {len(self.dataset)} queries"
        )

    def _filter_dataset(self) -> List[Dict]:
        """Apply category/difficulty filters and quick mode."""
        dataset = EVALUATION_DATASET

        if self.categories:
            dataset = [q for q in dataset if q["category"] in self.categories]

        if self.difficulties:
            dataset = [q for q in dataset if q["difficulty"] in self.difficulties]

        if self.quick:
            dataset = dataset[:5]

        return dataset

    def run(self) -> pd.DataFrame:
        """Execute the full experiment. Returns results DataFrame."""
        logger.info("=" * 70)
        logger.info("STARTING PV-RAG COMPARATIVE EXPERIMENT")
        logger.info("=" * 70)

        total_start = time.time()

        # Instantiate all systems
        logger.info("Initializing systems...")
        for name in self.system_names:
            try:
                self.system_instances[name] = get_system(name)
                logger.info(f"  ✓ {name}: {self.system_instances[name].description}")
            except Exception as e:
                logger.error(f"  ✗ {name}: {e}")

        # Run each query for each system
        total_queries = len(self.dataset) * len(self.system_instances)
        completed = 0

        for qi, query_entry in enumerate(self.dataset):
            qid = query_entry["id"]
            query = query_entry["query"]
            query_year = query_entry.get("expected_year")

            logger.info(f"\n{'─'*50}")
            logger.info(f"Query [{qi+1}/{len(self.dataset)}] {qid}: {query[:60]}...")

            for sys_name, system in self.system_instances.items():
                completed += 1
                logger.info(f"  System: {sys_name} ({completed}/{total_queries})")

                result = self._run_single(system, query_entry)
                self.raw_results.append(result)

                # Rate limiting for LLM
                if sys_name != "LLMOnly":  # LLMOnly doesn't use retrieval
                    time.sleep(RATE_LIMIT_DELAY)

        total_time = time.time() - total_start
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT COMPLETE: {completed} evaluations in {total_time:.1f}s")
        logger.info(f"{'='*70}")

        # Build results DataFrame
        df = pd.DataFrame(self.raw_results)
        self._save_results(df)
        return df

    def _run_single(self, system, query_entry: Dict) -> Dict:
        """Run a single system on a single query and compute metrics."""
        query = query_entry["query"]
        query_year = query_entry.get("expected_year")
        qid = query_entry["id"]

        result = {
            "query_id": qid,
            "query": query,
            "query_type": query_entry["query_type"],
            "expected_year": query_year,
            "category": query_entry["category"],
            "difficulty": query_entry["difficulty"],
            "system": system.name,
            "gold_law": query_entry.get("gold_law", ""),
            "gold_section": query_entry.get("gold_section", ""),
        }

        # 1. Retrieval
        t0 = time.time()
        try:
            rules = system.retrieve(query, query_year, limit=10)
        except Exception as e:
            logger.error(f"    Retrieval failed: {e}")
            rules = []
        retrieval_time = time.time() - t0

        result["n_retrieved"] = len(rules)
        result["retrieval_time"] = retrieval_time

        # Log top retrieval
        if rules:
            top = rules[0]
            logger.info(
                f"    Top: {top.get('law_name', '?')}, {top.get('section', '?')} "
                f"({top.get('start_year', '?')}-{top.get('end_year', '?')})"
            )

        # 2. Answer generation
        t1 = time.time()
        try:
            answer, confidence = system.generate_answer(query, rules, query_year)
        except Exception as e:
            logger.error(f"    Answer generation failed: {e}")
            answer = "Error generating answer."
            confidence = 0.0
        generation_time = time.time() - t1

        total_time = retrieval_time + generation_time

        result["answer"] = answer[:500]  # Truncate for storage
        result["confidence"] = confidence
        result["generation_time"] = generation_time
        result["total_time"] = total_time

        # 3. Compute metrics
        metrics = compute_all_metrics(
            query_entry=query_entry,
            retrieved_rules=rules,
            answer=answer,
            confidence=confidence,
            latency=total_time,
        )
        result.update(metrics)

        logger.info(
            f"    P@1={metrics['precision_at_1']:.2f} "
            f"TP@1={metrics['temporal_precision_at_1']:.2f} "
            f"KHR={metrics['keyword_hit_rate']:.2f} "
            f"THR={metrics['temporal_hallucination_rate']:.2f} "
            f"({total_time:.1f}s)"
        )

        return result

    def _save_results(self, df: pd.DataFrame):
        """Save results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_path = os.path.join(RESULTS_DIR, f"experiment_results_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved: {csv_path}")

        # JSON (compact)
        json_path = os.path.join(RESULTS_DIR, f"experiment_results_{timestamp}.json")
        df.to_json(json_path, orient="records", indent=2)
        logger.info(f"Results saved: {json_path}")

        # Summary tables
        summary = self._build_summary(df)
        summary_path = os.path.join(RESULTS_DIR, f"experiment_summary_{timestamp}.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info(f"Summary saved: {summary_path}")

        # Print with safe encoding for Windows console
        try:
            print("\n" + summary)
        except UnicodeEncodeError:
            print("\n" + summary.encode("ascii", errors="replace").decode("ascii"))

    def _build_summary(self, df: pd.DataFrame) -> str:
        """Build human-readable summary tables."""
        lines = []
        lines.append("=" * 90)
        lines.append("PV-RAG COMPARATIVE EXPERIMENT — RESULTS SUMMARY")
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"Systems: {', '.join(self.system_names)}")
        lines.append(f"Queries: {len(self.dataset)}")
        lines.append("=" * 90)

        # ── TABLE 1: Overall Metrics by System ──
        lines.append("\n" + "─" * 90)
        lines.append("TABLE 1: OVERALL METRICS BY SYSTEM")
        lines.append("─" * 90)

        metric_cols = [
            "precision_at_1", "precision_at_5", "recall_at_5", "mrr", "ndcg_at_5",
            "temporal_precision_at_1", "temporal_precision_at_5",
            "version_discrimination", "amendment_awareness",
            "temporal_hallucination_rate",
            "keyword_hit_rate", "answer_completeness",
            "latency_seconds",
        ]

        header = f"{'System':<20}"
        for col in metric_cols:
            short = self._short_name(col)
            header += f" {short:>7}"
        lines.append(header)
        lines.append("-" * len(header))

        for sys_name in sorted(df["system"].unique()):
            sdf = df[df["system"] == sys_name]
            row = f"{sys_name:<20}"
            for col in metric_cols:
                if col in sdf.columns:
                    val = sdf[col].dropna().mean()
                    row += f" {val:>7.3f}"
                else:
                    row += f" {'N/A':>7}"
            lines.append(row)

        # ── TABLE 2: Metrics by Category ──
        lines.append("\n" + "─" * 90)
        lines.append("TABLE 2: KEY METRICS BY QUERY CATEGORY")
        lines.append("─" * 90)

        key_metrics = ["precision_at_1", "temporal_precision_at_1", "keyword_hit_rate", "temporal_hallucination_rate"]

        for category in sorted(df["category"].unique()):
            lines.append(f"\n  Category: {category}")
            cdf = df[df["category"] == category]

            header2 = f"  {'System':<20}"
            for col in key_metrics:
                header2 += f" {self._short_name(col):>8}"
            header2 += f" {'N':>4}"
            lines.append(header2)
            lines.append("  " + "-" * (len(header2) - 2))

            for sys_name in sorted(cdf["system"].unique()):
                sdf = cdf[cdf["system"] == sys_name]
                row2 = f"  {sys_name:<20}"
                for col in key_metrics:
                    if col in sdf.columns:
                        val = sdf[col].dropna().mean()
                        row2 += f" {val:>8.3f}"
                    else:
                        row2 += f" {'N/A':>8}"
                row2 += f" {len(sdf):>4}"
                lines.append(row2)

        # ── TABLE 3: Metrics by Difficulty ──
        lines.append("\n" + "─" * 90)
        lines.append("TABLE 3: KEY METRICS BY DIFFICULTY")
        lines.append("─" * 90)

        for diff in ["easy", "medium", "hard"]:
            ddf = df[df["difficulty"] == diff]
            if ddf.empty:
                continue
            lines.append(f"\n  Difficulty: {diff}")

            header3 = f"  {'System':<20}"
            for col in key_metrics:
                header3 += f" {self._short_name(col):>8}"
            lines.append(header3)
            lines.append("  " + "-" * (len(header3) - 2))

            for sys_name in sorted(ddf["system"].unique()):
                sdf = ddf[ddf["system"] == sys_name]
                row3 = f"  {sys_name:<20}"
                for col in key_metrics:
                    if col in sdf.columns:
                        val = sdf[col].dropna().mean()
                        row3 += f" {val:>8.3f}"
                    else:
                        row3 += f" {'N/A':>8}"
                lines.append(row3)

        # ── TABLE 4: Statistical Significance (Win/Tie/Loss) ──
        lines.append("\n" + "─" * 90)
        lines.append("TABLE 4: PV-RAG vs BASELINES — WIN/TIE/LOSS")
        lines.append("─" * 90)

        pvrag_df = df[df["system"] == "PV-RAG"]
        if not pvrag_df.empty:
            for sys_name in sorted(df["system"].unique()):
                if sys_name == "PV-RAG":
                    continue
                other_df = df[df["system"] == sys_name]

                wins, ties, losses = 0, 0, 0
                for qid in pvrag_df["query_id"].unique():
                    pv_row = pvrag_df[pvrag_df["query_id"] == qid]
                    ot_row = other_df[other_df["query_id"] == qid]
                    if pv_row.empty or ot_row.empty:
                        continue

                    # Compare on keyword_hit_rate (primary quality metric)
                    pv_score = pv_row["keyword_hit_rate"].values[0]
                    ot_score = ot_row["keyword_hit_rate"].values[0]

                    if pv_score > ot_score + 0.05:
                        wins += 1
                    elif ot_score > pv_score + 0.05:
                        losses += 1
                    else:
                        ties += 1

                total = wins + ties + losses
                lines.append(
                    f"  PV-RAG vs {sys_name:<20}: "
                    f"W={wins:>2} T={ties:>2} L={losses:>2} "
                    f"(Win%={100 * wins / max(total, 1):.0f}%)"
                )

        # ── KEY FINDINGS ──
        lines.append("\n" + "─" * 90)
        lines.append("KEY FINDINGS")
        lines.append("─" * 90)

        if not pvrag_df.empty:
            pv_tp1 = pvrag_df["temporal_precision_at_1"].dropna().mean()
            lines.append(f"  • PV-RAG Temporal Precision@1: {pv_tp1:.3f}")

            for sys_name in sorted(df["system"].unique()):
                if sys_name == "PV-RAG":
                    continue
                other_tp1 = df[df["system"] == sys_name]["temporal_precision_at_1"].dropna().mean()
                improvement = ((pv_tp1 - other_tp1) / max(other_tp1, 0.001)) * 100
                lines.append(f"    vs {sys_name}: {improvement:+.1f}% improvement")

        lines.append("\n" + "=" * 90)
        return "\n".join(lines)

    @staticmethod
    def _short_name(metric: str) -> str:
        """Abbreviate metric names for table display."""
        mapping = {
            "precision_at_1": "P@1",
            "precision_at_3": "P@3",
            "precision_at_5": "P@5",
            "recall_at_5": "R@5",
            "recall_at_10": "R@10",
            "mrr": "MRR",
            "ndcg_at_5": "NDCG@5",
            "temporal_precision_at_1": "TP@1",
            "temporal_precision_at_3": "TP@3",
            "temporal_precision_at_5": "TP@5",
            "version_discrimination": "VDS",
            "amendment_awareness": "AAS",
            "temporal_hallucination_rate": "THR",
            "cross_era_accuracy": "CEA",
            "keyword_hit_rate": "KHR",
            "answer_completeness": "AC",
            "confidence_score": "Conf",
            "latency_seconds": "Lat(s)",
        }
        return mapping.get(metric, metric[:7])


# ──────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PV-RAG Comparative Experiment Runner"
    )
    parser.add_argument(
        "--systems", nargs="+",
        help="Systems to evaluate (default: all)",
        choices=list(ALL_SYSTEMS.keys()),
    )
    parser.add_argument(
        "--categories", nargs="+",
        help="Filter by query categories",
    )
    parser.add_argument(
        "--difficulties", nargs="+",
        help="Filter by difficulty levels",
        choices=["easy", "medium", "hard"],
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: only 5 queries",
    )
    args = parser.parse_args()

    runner = ExperimentRunner(
        system_names=args.systems,
        categories=args.categories,
        difficulties=args.difficulties,
        quick=args.quick,
    )
    df = runner.run()
    return df


if __name__ == "__main__":
    main()
