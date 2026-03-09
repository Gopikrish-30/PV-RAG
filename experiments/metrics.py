"""
PV-RAG Evaluation Metrics — Novel Temporal Legal QA Metrics

Includes:
 - Standard IR: Precision@K, Recall@K, MRR, NDCG@K
 - Novel Temporal: Temporal Precision@K, Version Discrimination Score,
   Amendment Awareness, Temporal Hallucination Rate
 - Answer Quality: Keyword Hit Rate, BERTScore (optional), Answer Completeness
 - Confidence Calibration: ECE, reliability diagram data
 - Latency: response time tracking
"""
import re
import time
import math
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# 1. STANDARD INFORMATION RETRIEVAL METRICS
# ──────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved_laws: List[str], gold_law: str, k: int = 5) -> float:
    """Fraction of top-K retrieved results matching the gold law."""
    top_k = retrieved_laws[:k]
    if not top_k:
        return 0.0
    matches = sum(1 for law in top_k if _law_match(law, gold_law))
    return matches / len(top_k)


def recall_at_k(retrieved_laws: List[str], gold_law: str, k: int = 10) -> float:
    """Whether the gold law appears at all in top-K (binary recall)."""
    top_k = retrieved_laws[:k]
    return 1.0 if any(_law_match(law, gold_law) for law in top_k) else 0.0


def mrr(retrieved_laws: List[str], gold_law: str) -> float:
    """Mean Reciprocal Rank — 1/rank of first correct result."""
    for i, law in enumerate(retrieved_laws):
        if _law_match(law, gold_law):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_laws: List[str], gold_law: str, k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    top_k = retrieved_laws[:k]
    dcg = 0.0
    for i, law in enumerate(top_k):
        rel = 1.0 if _law_match(law, gold_law) else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: all relevant at top
    n_relevant = sum(1 for law in retrieved_laws if _law_match(law, gold_law))
    ideal_rels = min(n_relevant, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────
# 2. NOVEL TEMPORAL METRICS (Key contribution of PV-RAG)
# ──────────────────────────────────────────────────────────────────────

def temporal_precision_at_k(
    retrieved_rules: List[Dict],
    query_year: int,
    k: int = 5,
) -> float:
    """
    NOVEL METRIC: Temporal Precision@K

    Fraction of top-K results whose [start_year, end_year] contains query_year.
    This measures whether the retrieval system returns *temporally valid* results,
    not just semantically relevant ones.

    A system with high semantic precision but low temporal precision returns
    the RIGHT law but the WRONG version.
    """
    top_k = retrieved_rules[:k]
    if not top_k or query_year is None:
        return 0.0

    valid = 0
    for rule in top_k:
        start = rule.get("start_year", 0)
        end = rule.get("end_year", 9999)
        if start <= query_year <= end:
            valid += 1

    return valid / len(top_k)


def version_discrimination_score(
    retrieved_rules: List[Dict],
    gold_temporal_range: Tuple[int, int],
    query_year: int,
    k: int = 5,
) -> float:
    """
    NOVEL METRIC: Version Discrimination Score (VDS)

    Measures the system's ability to pick the CORRECT version of a law
    when multiple versions exist. Returns 1.0 if the top-1 result exactly
    matches the gold temporal range, decaying with rank.

    VDS = max(1/rank_of_correct_version, 0) for exact range match
    Partial credit: 0.5 if the rule is temporally valid but wrong range
    """
    if gold_temporal_range is None or query_year is None:
        return 0.0

    gold_start, gold_end = gold_temporal_range

    for i, rule in enumerate(retrieved_rules[:k]):
        start = rule.get("start_year", 0)
        end = rule.get("end_year", 9999)

        # Exact version match
        if start == gold_start and (end == gold_end or (gold_end == 9999 and end >= 9999)):
            return 1.0 / (i + 1)

        # Temporally valid but different version
        if start <= query_year <= end:
            return 0.5 / (i + 1)

    return 0.0


def amendment_awareness_score(
    retrieved_rules: List[Dict],
    gold_law: str,
    gold_section: Optional[str],
) -> float:
    """
    NOVEL METRIC: Amendment Awareness Score (AAS)

    Measures whether the retrieval found multiple versions (amendments) of
    the same law+section. A higher score means the system surfaced the
    amendment history, not just one snapshot.

    AAS = min(n_unique_versions / 2, 1.0)
    (Normalized: 2+ versions = full score)
    """
    if not gold_law:
        return 0.0

    versions_seen = set()
    for rule in retrieved_rules:
        law = rule.get("law_name", "")
        section = rule.get("section", "")
        if _law_match(law, gold_law) and (not gold_section or _section_match(section, gold_section)):
            key = (rule.get("start_year", 0), rule.get("end_year", 9999))
            versions_seen.add(key)

    return min(len(versions_seen) / 2.0, 1.0)


def temporal_hallucination_rate(
    answer: str,
    query_year: Optional[int],
    gold_temporal_range: Optional[Tuple[int, int]],
) -> float:
    """
    NOVEL METRIC: Temporal Hallucination Rate (THR)

    Detects temporal inconsistencies in the generated answer.
    Checks if the answer mentions years outside the valid range
    or contradicts the query timeframe.

    Returns 0.0 (no hallucination) to 1.0 (severe hallucination).
    """
    if not answer or query_year is None:
        return 0.0

    hallucination_signals = 0
    total_checks = 0

    # Check 1: Does the answer mention the correct era?
    years_mentioned = [int(y) for y in re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', answer)]
    if years_mentioned and gold_temporal_range:
        total_checks += 1
        gold_start, gold_end = gold_temporal_range
        # If the answer ONLY mentions years outside the valid range, it's hallucinating
        valid_mentions = [y for y in years_mentioned if gold_start <= y <= min(gold_end, 2026)]
        if not valid_mentions and years_mentioned:
            hallucination_signals += 1

    # Check 2: Does the answer say "current" or "currently" for historical queries?
    if query_year and query_year < 2024:
        total_checks += 1
        current_words = re.findall(r'\b(?:current(?:ly)?|present(?:ly)?|as of today|now)\b', answer, re.I)
        if current_words:
            hallucination_signals += 0.5

    # Check 3: Does the answer reference a law that didn't exist at query time?
    total_checks += 1
    if query_year and query_year < 2024:
        if re.search(r'bharatiya nyaya sanhita|BNS|BNSS', answer, re.I):
            hallucination_signals += 1  # BNS didn't exist before 2024

    return hallucination_signals / max(total_checks, 1)


def cross_era_accuracy(
    retrieved_rules: List[Dict],
    answer: str,
    query_year: int,
    gold_law: str,
) -> float:
    """
    NOVEL METRIC: Cross-Era Accuracy (CEA)

    For queries spanning law replacement boundaries (IPC→BNS in 2024),
    measures whether the system correctly identifies which law applies.

    Returns 1.0 if the answer/top retrieval references the correct-era law.
    """
    if not gold_law or not query_year:
        return 0.0

    # Check answer text
    answer_lower = answer.lower() if answer else ""
    gold_lower = gold_law.lower()

    # Check if gold law name (or key part) appears in answer
    gold_parts = [p.strip().lower() for p in gold_lower.split(",")[0].split() if len(p) > 3]
    answer_has_gold = any(part in answer_lower for part in gold_parts)

    # Check top retrievals
    retrieval_has_gold = any(
        _law_match(r.get("law_name", ""), gold_law)
        for r in retrieved_rules[:3]
    )

    if answer_has_gold and retrieval_has_gold:
        return 1.0
    elif answer_has_gold or retrieval_has_gold:
        return 0.5
    return 0.0


# ──────────────────────────────────────────────────────────────────────
# 3. ANSWER QUALITY METRICS
# ──────────────────────────────────────────────────────────────────────

def keyword_hit_rate(answer: str, gold_keywords: List[str]) -> float:
    """Fraction of gold keywords present in the answer (case-insensitive)."""
    if not answer or not gold_keywords:
        return 0.0

    answer_lower = answer.lower()
    hits = sum(1 for kw in gold_keywords if kw.lower() in answer_lower)
    return hits / len(gold_keywords)


def answer_completeness(
    answer: str,
    gold_law: str,
    gold_section: Optional[str],
    gold_keywords: List[str],
) -> float:
    """
    Composite answer quality score combining:
    - Law citation (0.3 weight)
    - Section citation (0.2 weight)
    - Keyword coverage (0.5 weight)
    """
    if not answer:
        return 0.0

    answer_lower = answer.lower()

    # Law citation
    law_score = 0.0
    if gold_law:
        gold_parts = [p.strip().lower() for p in gold_law.split(",")[0].split() if len(p) > 3]
        if any(p in answer_lower for p in gold_parts):
            law_score = 1.0

    # Section citation
    section_score = 0.0
    if gold_section:
        sec_num = re.findall(r'\d+[A-Za-z]*', gold_section)
        if sec_num and any(s in answer_lower for s in sec_num):
            section_score = 1.0
    else:
        section_score = 1.0  # N/A → full credit

    # Keyword coverage
    kw_score = keyword_hit_rate(answer, gold_keywords)

    return 0.3 * law_score + 0.2 * section_score + 0.5 * kw_score


# ──────────────────────────────────────────────────────────────────────
# 4. CONFIDENCE CALIBRATION METRICS
# ──────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: List[float],
    correctness: List[float],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures how well confidence scores correlate with actual correctness.
    Lower is better. Perfect calibration = 0.0.
    """
    if not confidences or not correctness:
        return 1.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = [(lo <= c < hi) for c in confidences]
        bin_size = sum(mask)

        if bin_size > 0:
            avg_conf = np.mean([c for c, m in zip(confidences, mask) if m])
            avg_acc = np.mean([a for a, m in zip(correctness, mask) if m])
            ece += (bin_size / total) * abs(avg_acc - avg_conf)

    return ece


def reliability_diagram_data(
    confidences: List[float],
    correctness: List[float],
    n_bins: int = 10,
) -> Dict:
    """Generate data for reliability diagram plotting."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = [(lo <= c < hi) for c in confidences]
        bin_size = sum(mask)
        center = (lo + hi) / 2

        if bin_size > 0:
            avg_acc = np.mean([a for a, m in zip(correctness, mask) if m])
            bin_centers.append(center)
            bin_accuracies.append(avg_acc)
            bin_counts.append(bin_size)

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


# ──────────────────────────────────────────────────────────────────────
# 5. AGGREGATE METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    query_entry: Dict,
    retrieved_rules: List[Dict],
    answer: str,
    confidence: float,
    latency: float,
) -> Dict:
    """
    Compute ALL metrics for a single query-answer pair.

    Returns a flat dict with all metric values.
    """
    gold_law = query_entry.get("gold_law", "")
    gold_section = query_entry.get("gold_section")
    gold_keywords = query_entry.get("gold_answer_keywords", [])
    gold_range = query_entry.get("gold_temporal_range")
    query_year = query_entry.get("expected_year")

    retrieved_laws = [r.get("law_name", "") for r in retrieved_rules]

    metrics = {
        # IR metrics
        "precision_at_1": precision_at_k(retrieved_laws, gold_law, k=1),
        "precision_at_3": precision_at_k(retrieved_laws, gold_law, k=3),
        "precision_at_5": precision_at_k(retrieved_laws, gold_law, k=5),
        "recall_at_5": recall_at_k(retrieved_laws, gold_law, k=5),
        "recall_at_10": recall_at_k(retrieved_laws, gold_law, k=10),
        "mrr": mrr(retrieved_laws, gold_law),
        "ndcg_at_5": ndcg_at_k(retrieved_laws, gold_law, k=5),

        # Temporal metrics (NOVEL)
        "temporal_precision_at_1": temporal_precision_at_k(retrieved_rules, query_year, k=1),
        "temporal_precision_at_3": temporal_precision_at_k(retrieved_rules, query_year, k=3),
        "temporal_precision_at_5": temporal_precision_at_k(retrieved_rules, query_year, k=5),
        "version_discrimination": version_discrimination_score(
            retrieved_rules, gold_range, query_year, k=5
        ),
        "amendment_awareness": amendment_awareness_score(
            retrieved_rules, gold_law, gold_section
        ),
        "temporal_hallucination_rate": temporal_hallucination_rate(
            answer, query_year, gold_range
        ),
        "cross_era_accuracy": cross_era_accuracy(
            retrieved_rules, answer, query_year, gold_law
        ) if query_year else None,

        # Answer quality
        "keyword_hit_rate": keyword_hit_rate(answer, gold_keywords),
        "answer_completeness": answer_completeness(
            answer, gold_law, gold_section, gold_keywords
        ),

        # Confidence
        "confidence_score": confidence,

        # Latency
        "latency_seconds": latency,
    }

    return metrics


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def _law_match(retrieved_law: str, gold_law: str) -> bool:
    """Fuzzy match between retrieved and gold law names."""
    if not retrieved_law or not gold_law:
        return False

    r = retrieved_law.lower().strip()
    g = gold_law.lower().strip()

    # Exact match
    if r == g:
        return True

    # Substring match (either direction)
    if g in r or r in g:
        return True

    # Key-term match (ignore year, commas)
    r_terms = set(re.findall(r'[a-z]+', r)) - {"act", "the", "of", "and", "for"}
    g_terms = set(re.findall(r'[a-z]+', g)) - {"act", "the", "of", "and", "for"}

    if len(r_terms & g_terms) >= 2:
        return True

    return False


def _section_match(retrieved_section: str, gold_section: str) -> bool:
    """Match section identifiers (e.g., 'Sec 129' matches '129', 'Section 129')."""
    r_nums = set(re.findall(r'\d+[A-Za-z]*', str(retrieved_section)))
    g_nums = set(re.findall(r'\d+[A-Za-z]*', str(gold_section)))
    return bool(r_nums & g_nums)
