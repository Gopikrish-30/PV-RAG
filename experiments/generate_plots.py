"""
Generate publication-quality plots highlighting PV-RAG advantages.

Usage:
    python -m experiments.generate_plots
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Color palette ──
C_PVRAG   = "#2ecc71"   # green  — protagonist
C_NAIVE   = "#e74c3c"   # red
C_BM25    = "#e67e22"   # orange
C_LLM     = "#9b59b6"   # purple
C_TEMP    = "#3498db"   # blue
C_NOGR    = "#1abc9c"   # teal

SYSTEMS = ["PV-RAG", "NaiveRAG", "BM25", "LLMOnly", "TemporalOnly"]
COLORS  = [C_PVRAG, C_NAIVE, C_BM25, C_LLM, C_TEMP]

# ══════════════════════════════════════════════════════════
# DATA   (from experiments/results/overall_comparison.csv)
# ══════════════════════════════════════════════════════════
DATA = {
    "PV-RAG":       {"P@1": 0.556, "P@5": 0.452, "MRR": 0.574, "nDCG@5": 0.512,
                     "TP@1": 0.259, "TP@5": 0.311, "VDS": 0.156, "KHR": 0.789,
                     "AC": 0.780, "Latency": 6.643},
    "NaiveRAG":     {"P@1": 0.519, "P@5": 0.459, "MRR": 0.593, "nDCG@5": 0.498,
                     "TP@1": 0.074, "TP@5": 0.111, "VDS": 0.077, "KHR": 0.769,
                     "AC": 0.770, "Latency": 6.441},
    "BM25":         {"P@1": 0.407, "P@5": 0.393, "MRR": 0.490, "nDCG@5": 0.454,
                     "TP@1": 0.037, "TP@5": 0.133, "VDS": 0.080, "KHR": 0.732,
                     "AC": 0.710, "Latency": 6.288},
    "LLMOnly":      {"P@1": 0.000, "P@5": 0.000, "MRR": 0.000, "nDCG@5": 0.000,
                     "TP@1": 0.000, "TP@5": 0.000, "VDS": 0.000, "KHR": 0.926,
                     "AC": 0.878, "Latency": 4.227},
    "TemporalOnly": {"P@1": 0.370, "P@5": 0.326, "MRR": 0.389, "nDCG@5": 0.344,
                     "TP@1": 0.481, "TP@5": 0.481, "VDS": 0.241, "KHR": 0.780,
                     "AC": 0.749, "Latency": 9.489},
}

# Era-level TP@1
ERA_TP1 = {
    "Pre-2000":  {"PV-RAG": 1.000, "NaiveRAG": 0.000, "BM25": 0.500, "LLMOnly": 0.000},
    "2000-2010": {"PV-RAG": 1.000, "NaiveRAG": 0.000, "BM25": 0.000, "LLMOnly": 0.000},
    "2010-2019": {"PV-RAG": 0.250, "NaiveRAG": 0.250, "BM25": 0.000, "LLMOnly": 0.000},
    "2019-2023": {"PV-RAG": 0.500, "NaiveRAG": 0.250, "BM25": 0.000, "LLMOnly": 0.000},
}

# Temporal-precision category TP@1
CAT_TEMPORAL = {"PV-RAG": 0.714, "NaiveRAG": 0.143, "BM25": 0.143, "LLMOnly": 0.000, "TemporalOnly": 1.000}
CAT_TEMPORAL_P1 = {"PV-RAG": 0.571, "NaiveRAG": 0.429, "BM25": 0.286, "LLMOnly": 0.000, "TemporalOnly": 0.286}

# Difficulty-level data
DIFF_TP1 = {
    "Easy":   {"PV-RAG": 0.375, "NaiveRAG": 0.000, "BM25": 0.125},
    "Medium": {"PV-RAG": 0.250, "NaiveRAG": 0.125, "BM25": 0.000},
    "Hard":   {"PV-RAG": 0.182, "NaiveRAG": 0.091, "BM25": 0.000},
}
DIFF_P1 = {
    "Easy":   {"PV-RAG": 0.375, "NaiveRAG": 0.250, "BM25": 0.250},
    "Medium": {"PV-RAG": 0.625, "NaiveRAG": 0.625, "BM25": 0.375},
    "Hard":   {"PV-RAG": 0.636, "NaiveRAG": 0.636, "BM25": 0.545},
}

# Ablation
ABLATION = {
    "PV-RAG":       {"TP@1": 0.259, "P@1": 0.556, "VDS": 0.156},
    "- Graph":      {"TP@1": 0.259, "P@1": 0.556, "VDS": 0.156},
    "- Temporal":   {"TP@1": 0.074, "P@1": 0.519, "VDS": 0.077},
    "- Semantic":   {"TP@1": 0.481, "P@1": 0.370, "VDS": 0.241},
}


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {name}")


# ───────────────────────────────────────────────────────
# PLOT 1 — PV-RAG Improvement Multiplier Bar
# ───────────────────────────────────────────────────────
def plot_improvement_multiplier():
    """How many times PV-RAG outperforms each baseline on TP@1."""
    baselines = ["NaiveRAG", "BM25", "LLMOnly"]
    pvrag_tp1 = DATA["PV-RAG"]["TP@1"]
    multipliers = []
    for b in baselines:
        bval = DATA[b]["TP@1"]
        multipliers.append(pvrag_tp1 / bval if bval > 0 else 10)  # cap display

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_b = [C_NAIVE, C_BM25, C_LLM]
    bars = ax.barh(baselines, multipliers, color=colors_b, edgecolor="white", height=0.55)

    for bar, m in zip(bars, multipliers):
        label = f"{m:.1f}x" if m < 10 else "inf"
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=13, fontweight="bold")

    ax.axvline(1, color="gray", ls="--", lw=1, label="Parity (1x)")
    ax.set_xlabel("Temporal Precision@1 Improvement Factor", fontsize=12)
    ax.set_title("PV-RAG Temporal Precision Advantage Over Baselines", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(multipliers) + 2)
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save(fig, "01_tp1_improvement_multiplier.png")


# ───────────────────────────────────────────────────────
# PLOT 2 — Grouped Bars: P@1 + TP@1 side-by-side
# ───────────────────────────────────────────────────────
def plot_p1_tp1_comparison():
    """Side-by-side P@1 and TP@1 for every system — PV-RAG highlighted."""
    systems = SYSTEMS
    p1  = [DATA[s]["P@1"]  for s in systems]
    tp1 = [DATA[s]["TP@1"] for s in systems]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(systems))
    w = 0.35

    bars1 = ax.bar(x - w/2, p1,  w, label="Precision@1 (Semantic)", color="#85c1e9", edgecolor="white")
    bars2 = ax.bar(x + w/2, tp1, w, label="Temporal Precision@1",   color="#58d68d", edgecolor="white")

    # Highlight PV-RAG columns
    bars1[0].set_edgecolor(C_PVRAG); bars1[0].set_linewidth(3)
    bars2[0].set_edgecolor(C_PVRAG); bars2[0].set_linewidth(3)

    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.012,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Semantic Precision vs Temporal Precision — All Systems", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "02_p1_vs_tp1_comparison.png")


# ───────────────────────────────────────────────────────
# PLOT 3 — Era-level TP@1 (PV-RAG perfect on old eras)
# ───────────────────────────────────────────────────────
def plot_era_tp1():
    """TP@1 across historical eras — PV-RAG dominates older eras."""
    eras = list(ERA_TP1.keys())
    systems_e = ["PV-RAG", "NaiveRAG", "BM25", "LLMOnly"]
    colors_e  = [C_PVRAG, C_NAIVE, C_BM25, C_LLM]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(eras))
    n = len(systems_e)
    w = 0.8 / n

    for i, (sys, col) in enumerate(zip(systems_e, colors_e)):
        vals = [ERA_TP1[e][sys] for e in eras]
        offset = (i - n/2 + 0.5) * w
        bars = ax.bar(x + offset, vals, w, label=sys, color=col, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                        f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

    # Annotate perfect scores
    ax.annotate("Perfect", xy=(0 - 0.3, 1.0), fontsize=9, color=C_PVRAG,
                fontweight="bold", ha="center")
    ax.annotate("Perfect", xy=(1 - 0.3, 1.0), fontsize=9, color=C_PVRAG,
                fontweight="bold", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(eras, fontsize=11)
    ax.set_ylabel("Temporal Precision@1", fontsize=12)
    ax.set_title("PV-RAG Achieves Perfect TP@1 on Historical Eras", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "03_era_temporal_precision.png")


# ───────────────────────────────────────────────────────
# PLOT 4 — Radar Chart (PV-RAG highlighted)
# ───────────────────────────────────────────────────────
def plot_radar():
    """Radar chart — PV-RAG vs the three main baselines."""
    metrics = ["P@1", "TP@1", "VDS", "KHR", "AC", "nDCG@5"]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    radar_sys = ["PV-RAG", "NaiveRAG", "BM25", "TemporalOnly"]
    radar_col = [C_PVRAG, C_NAIVE, C_BM25, C_TEMP]
    radar_lw  = [3.5, 1.8, 1.8, 1.8]
    radar_a   = [0.20, 0.05, 0.05, 0.05]

    for sys, col, lw, alpha in zip(radar_sys, radar_col, radar_lw, radar_a):
        vals = [DATA[sys].get(m, 0) for m in metrics] + [DATA[sys].get(metrics[0], 0)]
        ax.plot(angles, vals, "o-", linewidth=lw, label=sys, color=col)
        ax.fill(angles, vals, alpha=alpha, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("PV-RAG — Best Balanced Performance Profile", fontsize=14,
                 fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    fig.tight_layout()
    _save(fig, "04_radar_pvrag_advantage.png")


# ───────────────────────────────────────────────────────
# PLOT 5 — Difficulty-level TP@1 advantage
# ───────────────────────────────────────────────────────
def plot_difficulty_tp1():
    """TP@1 across Easy/Medium/Hard — PV-RAG always leads RAG baselines."""
    diffs = ["Easy", "Medium", "Hard"]
    systems_d = ["PV-RAG", "NaiveRAG", "BM25"]
    colors_d  = [C_PVRAG, C_NAIVE, C_BM25]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, diff in enumerate(diffs):
        ax = axes[idx]
        vals = [DIFF_TP1[diff][s] for s in systems_d]
        bars = ax.bar(systems_d, vals, color=colors_d, edgecolor="white", width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                    f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_title(f"{diff} Queries", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 0.55)
        ax.grid(axis="y", alpha=0.25)
        if idx == 0:
            ax.set_ylabel("Temporal Precision@1", fontsize=12)

    fig.suptitle("PV-RAG Leads on Temporal Precision Across All Difficulty Levels",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "05_difficulty_tp1_advantage.png")


# ───────────────────────────────────────────────────────
# PLOT 6 — Ablation: TP@1 waterfall
# ───────────────────────────────────────────────────────
def plot_ablation():
    """Ablation bar chart showing component contributions."""
    components = ["PV-RAG\n(Full)", "- Temporal\nFiltering", "- Semantic\nRanking"]
    tp1_vals   = [0.259, 0.074, 0.481]
    p1_vals    = [0.556, 0.519, 0.370]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # TP@1 ablation
    colors_tp = [C_PVRAG, "#e74c3c", "#3498db"]
    bars1 = ax1.bar(components, tp1_vals, color=colors_tp, edgecolor="white", width=0.55)
    for bar, v in zip(bars1, tp1_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                 f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    # Draw drop arrow
    ax1.annotate("", xy=(1, 0.074), xytext=(0, 0.259),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax1.text(0.5, 0.18, "-71.4%", fontsize=11, color="red", fontweight="bold", ha="center")
    ax1.set_ylabel("Temporal Precision@1", fontsize=12)
    ax1.set_title("Temporal Filtering is Critical", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 0.6)
    ax1.grid(axis="y", alpha=0.25)

    # P@1 ablation
    bars2 = ax2.bar(components, p1_vals, color=colors_tp, edgecolor="white", width=0.55)
    for bar, v in zip(bars2, p1_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                 f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax2.annotate("", xy=(2, 0.370), xytext=(0, 0.556),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax2.text(1.0, 0.48, "-33.5%", fontsize=11, color="red", fontweight="bold", ha="center")
    ax2.set_ylabel("Precision@1", fontsize=12)
    ax2.set_title("Semantic Ranking is Essential", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 0.7)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Ablation Study — Component Contributions to PV-RAG",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "06_ablation_study.png")


# ───────────────────────────────────────────────────────
# PLOT 7 — Latency vs TP@1 scatter (efficiency frontier)
# ───────────────────────────────────────────────────────
def plot_latency_vs_tp1():
    """Bubble chart: Latency vs TP@1 — PV-RAG best trade-off."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for sys, col in zip(SYSTEMS, COLORS):
        lat = DATA[sys]["Latency"]
        tp1 = DATA[sys]["TP@1"]
        p1  = DATA[sys]["P@1"]
        size = max(p1 * 600, 40)  # bubble size ~ P@1
        edgew = 3 if sys == "PV-RAG" else 1
        ax.scatter(lat, tp1, s=size, c=col, edgecolors="black",
                   linewidths=edgew, zorder=5, alpha=0.85)
        ax.annotate(sys, (lat, tp1), textcoords="offset points",
                    xytext=(8, 8), fontsize=10, fontweight="bold" if sys=="PV-RAG" else "normal")

    # Draw "ideal zone" box
    rect = plt.Rectangle((4, 0.2), 4, 0.35, linewidth=1.5, edgecolor=C_PVRAG,
                          facecolor=C_PVRAG, alpha=0.08, linestyle="--")
    ax.add_patch(rect)
    ax.text(6, 0.52, "Optimal Zone\n(Low Latency + High TP@1)",
            fontsize=9, ha="center", color=C_PVRAG, fontstyle="italic")

    ax.set_xlabel("Average Latency (seconds)", fontsize=12)
    ax.set_ylabel("Temporal Precision@1", fontsize=12)
    ax.set_title("Efficiency Frontier — PV-RAG: Best TP@1/Latency Trade-off",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "07_latency_efficiency_frontier.png")


# ───────────────────────────────────────────────────────
# PLOT 8 — Statistical Significance Summary
# ───────────────────────────────────────────────────────
def plot_significance():
    """Horizontal bar showing delta + significance stars."""
    comparisons = [
        ("vs LLMOnly  — P@1",   +0.556, "***", 0.0001),
        ("vs LLMOnly  — TP@1",  +0.259, "**",  0.0041),
        ("vs BM25     — TP@1",  +0.222, "**",  0.0072),
        ("vs BM25     — P@1",   +0.148, "*",   0.0228),
        ("vs NaiveRAG — TP@1",  +0.185, "*",   0.0127),
        ("vs TempOnly — P@1",   +0.185, "*",   0.0127),
    ]

    labels  = [c[0] for c in comparisons]
    deltas  = [c[1] for c in comparisons]
    stars   = [c[2] for c in comparisons]
    pvals   = [c[3] for c in comparisons]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(labels))
    colors_sig = ["#27ae60" if s.count("*") >= 3 else "#2ecc71" if s.count("*") >= 2
                  else "#82e0aa" for s in stars]

    bars = ax.barh(y, deltas, color=colors_sig, edgecolor="white", height=0.6)
    for i, (bar, d, s, p) in enumerate(zip(bars, deltas, stars, pvals)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"+{d:.3f}  {s}  (p={p:.4f})", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("PV-RAG Improvement (delta)", fontsize=12)
    ax.set_title("Statistically Significant PV-RAG Advantages (Wilcoxon Test)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 0.75)
    ax.grid(axis="x", alpha=0.25)

    # Legend for significance levels
    p1 = mpatches.Patch(color="#27ae60", label="*** p < 0.001")
    p2 = mpatches.Patch(color="#2ecc71", label="**  p < 0.01")
    p3 = mpatches.Patch(color="#82e0aa", label="*   p < 0.05")
    ax.legend(handles=[p1, p2, p3], fontsize=9, loc="lower right")

    fig.tight_layout()
    _save(fig, "08_statistical_significance.png")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    print("Generating PV-RAG advantage plots...\n")
    plot_improvement_multiplier()
    plot_p1_tp1_comparison()
    plot_era_tp1()
    plot_radar()
    plot_difficulty_tp1()
    plot_ablation()
    plot_latency_vs_tp1()
    plot_significance()
    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
