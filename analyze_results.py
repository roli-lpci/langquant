#!/usr/bin/env python3
"""
LangQuant — Information-Theoretic Analysis
Uses pyitlib, infomeasure, scipy.stats for proper measurement.

Analyzes:
1. Matrix run (575 trials): significance of scaffold effects, MI between condition and score
2. LPCI A/B test (40 turns): scaffold-output mutual information, KL divergence between conditions,
   significance of probe differences, transfer entropy (scaffold → output)
"""

import json
import re
from collections import Counter

import numpy as np
from scipy import stats
from pyitlib import discrete_random_variable as drv


# ── Helpers ──────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def token_distribution(text: str) -> np.ndarray:
    """Token frequency distribution as probability vector."""
    tokens = tokenize(text)
    if not tokens:
        return np.array([1.0])
    counts = Counter(tokens)
    total = sum(counts.values())
    return np.array([c / total for c in counts.values()])


def shannon_entropy(text: str) -> float:
    """Shannon entropy of token distribution in bits."""
    dist = token_distribution(text)
    return float(-np.sum(dist * np.log2(dist + 1e-12)))


def kl_divergence(p_text: str, q_text: str) -> float:
    """KL divergence D(P || Q) between token distributions of two texts.
    Uses shared vocabulary to align distributions."""
    p_tokens = tokenize(p_text)
    q_tokens = tokenize(q_text)
    if not p_tokens or not q_tokens:
        return 0.0

    # Build shared vocabulary
    all_tokens = set(p_tokens) | set(q_tokens)
    vocab = sorted(all_tokens)

    p_counts = Counter(p_tokens)
    q_counts = Counter(q_tokens)

    # Laplace smoothing
    p_dist = np.array([p_counts.get(t, 0) + 1 for t in vocab], dtype=float)
    q_dist = np.array([q_counts.get(t, 0) + 1 for t in vocab], dtype=float)

    p_dist /= p_dist.sum()
    q_dist /= q_dist.sum()

    return float(np.sum(p_dist * np.log2(p_dist / q_dist)))


def discretize(values: list[float], bins: int = 5) -> np.ndarray:
    """Discretize continuous values into bins for MI calculation."""
    arr = np.array(values)
    if arr.std() == 0:
        return np.zeros(len(arr), dtype=int)
    edges = np.linspace(arr.min() - 0.001, arr.max() + 0.001, bins + 1)
    return np.digitize(arr, edges[:-1]) - 1


# ── Analysis 1: Matrix Run ──────────────────────────────────────────────────

def analyze_matrix():
    print("=" * 70)
    print("ANALYSIS 1: MATRIX RUN (575 trials)")
    print("=" * 70)

    trials = []
    with open("results/full_run_v1.jsonl") as f:
        for line in f:
            trials.append(json.loads(line))

    # ── 1a. Significance: does condition affect score? (Kruskal-Wallis) ────
    print("\n## 1a. Kruskal-Wallis: Does scaffold condition affect task score?")
    conditions = {}
    for t in trials:
        c = t["condition"]
        if c not in conditions:
            conditions[c] = []
        conditions[c].append(t["task_score"])

    cond_names = sorted(conditions.keys())
    cond_scores = [conditions[c] for c in cond_names]

    h_stat, p_val = stats.kruskal(*cond_scores)
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_val:.6f}")
    print(f"  Significant at α=0.05: {'YES' if p_val < 0.05 else 'NO'}")

    for c in cond_names:
        scores = conditions[c]
        print(f"  {c:25s}: mean={np.mean(scores):.3f}  std={np.std(scores):.3f}  n={len(scores)}")

    # ── 1b. Pairwise Mann-Whitney: each condition vs naked ────────────────
    print("\n## 1b. Mann-Whitney U: Each condition vs naked")
    naked_scores = conditions.get("naked", [])
    for c in cond_names:
        if c == "naked":
            continue
        u_stat, p_val = stats.mannwhitneyu(naked_scores, conditions[c], alternative="two-sided")
        effect = np.mean(conditions[c]) - np.mean(naked_scores)
        print(f"  naked vs {c:25s}: U={u_stat:.0f}  p={p_val:.6f}  Δmean={effect:+.3f}  sig={'*' if p_val < 0.05 else 'ns'}")

    # ── 1c. Mutual Information: condition → score ─────────────────────────
    print("\n## 1c. Mutual Information: condition → task_score")
    cond_labels = np.array([t["condition"] for t in trials])
    # Encode conditions as integers
    cond_map = {c: i for i, c in enumerate(sorted(set(cond_labels)))}
    cond_ints = np.array([cond_map[c] for c in cond_labels])
    score_bins = discretize([t["task_score"] for t in trials], bins=5)

    mi = drv.information_mutual(cond_ints, score_bins)
    h_cond = drv.entropy(cond_ints)
    h_score = drv.entropy(score_bins)
    nmi = mi / min(h_cond, h_score) if min(h_cond, h_score) > 0 else 0

    print(f"  MI(condition; score) = {mi:.4f} bits")
    print(f"  H(condition) = {h_cond:.4f} bits")
    print(f"  H(score) = {h_score:.4f} bits")
    print(f"  Normalized MI = {nmi:.4f}")
    print(f"  Interpretation: condition explains {nmi*100:.1f}% of score variation (information-theoretic)")

    # ── 1d. MI: condition → behavioral complexity ─────────────────────────
    print("\n## 1d. Mutual Information: condition → behavioral_complexity")
    bc_bins = discretize([t["bc_composite_score"] for t in trials], bins=5)
    mi_bc = drv.information_mutual(cond_ints, bc_bins)
    h_bc = drv.entropy(bc_bins)
    nmi_bc = mi_bc / min(h_cond, h_bc) if min(h_cond, h_bc) > 0 else 0

    print(f"  MI(condition; BC) = {mi_bc:.4f} bits")
    print(f"  Normalized MI = {nmi_bc:.4f}")
    print(f"  Interpretation: condition explains {nmi_bc*100:.1f}% of BC variation")

    # ── 1e. MI: model_size → score (is model size more predictive?) ──────
    print("\n## 1e. Mutual Information: model_size → score (comparison)")
    model_map = {m: i for i, m in enumerate(sorted(set(t["model"] for t in trials)))}
    model_ints = np.array([model_map[t["model"]] for t in trials])
    mi_model = drv.information_mutual(model_ints, score_bins)
    h_model = drv.entropy(model_ints)
    nmi_model = mi_model / min(h_model, h_score) if min(h_model, h_score) > 0 else 0

    print(f"  MI(model; score) = {mi_model:.4f} bits")
    print(f"  Normalized MI = {nmi_model:.4f}")
    print(f"  Interpretation: model size explains {nmi_model*100:.1f}% of score variation")
    print(f"  → Condition explains {nmi*100:.1f}% vs model size explains {nmi_model*100:.1f}%")

    # ── 1f. Per-model: does condition matter more for small models? ───────
    print("\n## 1f. Condition effect by model size")
    for model in sorted(set(t["model"] for t in trials)):
        model_trials = [t for t in trials if t["model"] == model]
        if len(model_trials) < 20:
            continue
        m_conds = {}
        for t in model_trials:
            c = t["condition"]
            if c not in m_conds:
                m_conds[c] = []
            m_conds[c].append(t["task_score"])

        m_cond_scores = [m_conds[c] for c in sorted(m_conds.keys()) if len(m_conds[c]) > 2]
        if len(m_cond_scores) > 1:
            h, p = stats.kruskal(*m_cond_scores)
            print(f"  {model:15s}: Kruskal-Wallis H={h:.3f}  p={p:.6f}  sig={'*' if p < 0.05 else 'ns'}")


# ── Analysis 2: LPCI A/B Test ───────────────────────────────────────────────

def analyze_lpci():
    print(f"\n\n{'=' * 70}")
    print("ANALYSIS 2: LPCI A/B TEST (40 turns)")
    print("=" * 70)

    results = []
    with open("results/lpci_ab_test.jsonl") as f:
        for line in f:
            results.append(json.loads(line))

    naked = [r for r in results if r["condition"] == "naked"]
    compressed = [r for r in results if r["condition"] == "compressed"]

    # ── 2a. KL divergence between scaffold distributions per turn ────────
    print("\n## 2a. KL Divergence: scaffold distributions (naked vs compressed) per turn")
    print(f"  {'Turn':>4s}  {'D(N||C)':>8s}  {'D(C||N)':>8s}  {'Symmetric':>10s}")
    kl_values = []
    for i in range(20):
        n_scaffold = naked[i].get("scaffold_snapshot", "")
        c_scaffold = compressed[i].get("scaffold_snapshot", "")
        if n_scaffold and c_scaffold:
            d_nc = kl_divergence(n_scaffold, c_scaffold)
            d_cn = kl_divergence(c_scaffold, n_scaffold)
            sym = (d_nc + d_cn) / 2
            kl_values.append(sym)
            print(f"  {i+1:4d}  {d_nc:8.4f}  {d_cn:8.4f}  {sym:10.4f}")
        else:
            kl_values.append(0)

    print(f"\n  Mean symmetric KL: {np.mean(kl_values):.4f}")
    print(f"  Trend: {'diverging' if kl_values[-1] > kl_values[0] else 'converging'} (t1={kl_values[0]:.4f} → t20={kl_values[-1]:.4f})")

    # ── 2b. Scaffold entropy trajectory ──────────────────────────────────
    print("\n## 2b. Scaffold entropy trajectories")
    n_entropies = [r["scaffold_entropy"] for r in naked]
    c_entropies = [r["scaffold_entropy"] for r in compressed]

    # Mann-Whitney on entropy trajectories
    u, p = stats.mannwhitneyu(n_entropies, c_entropies, alternative="two-sided")
    print(f"  Naked:      mean={np.mean(n_entropies):.4f}  std={np.std(n_entropies):.4f}")
    print(f"  Compressed: mean={np.mean(c_entropies):.4f}  std={np.std(c_entropies):.4f}")
    print(f"  Mann-Whitney U={u:.0f}  p={p:.6f}  sig={'*' if p < 0.05 else 'ns'}")

    # ── 2c. MI: scaffold content → response content ──────────────────────
    print("\n## 2c. Mutual Information: scaffold → response (per condition)")
    for cond, data in [("naked", naked), ("compressed", compressed)]:
        scaffold_entropies = []
        response_entropies = []
        for r in data:
            s = r.get("scaffold_snapshot", "")
            resp = r.get("response", "")
            scaffold_entropies.append(shannon_entropy(s))
            response_entropies.append(shannon_entropy(resp))

        s_bins = discretize(scaffold_entropies, bins=4)
        r_bins = discretize(response_entropies, bins=4)
        mi = drv.information_mutual(s_bins, r_bins)
        h_s = drv.entropy(s_bins)
        h_r = drv.entropy(r_bins)
        nmi = mi / min(h_s, h_r) if min(h_s, h_r) > 0 else 0

        print(f"  {cond:12s}: MI(scaffold_entropy; response_entropy) = {mi:.4f} bits  NMI = {nmi:.4f}")

    # ── 2d. Scaffold-response token overlap (relational metric) ──────────
    print("\n## 2d. Scaffold → Response token overlap (relational)")
    print("  (What fraction of scaffold vocabulary appears in response?)")
    print(f"  {'Turn':>4s}  {'Naked overlap':>14s}  {'Compressed overlap':>18s}")
    n_overlaps = []
    c_overlaps = []
    for i in range(20):
        for cond, data, overlaps in [("naked", naked, n_overlaps), ("compressed", compressed, c_overlaps)]:
            scaffold_tokens = set(tokenize(data[i].get("scaffold_snapshot", "")))
            response_tokens = set(tokenize(data[i].get("response", "")))
            if scaffold_tokens:
                overlap = len(scaffold_tokens & response_tokens) / len(scaffold_tokens)
                overlaps.append(overlap)
            else:
                overlaps.append(0)

    for i in range(20):
        print(f"  {i+1:4d}  {n_overlaps[i]:14.3f}  {c_overlaps[i]:18.3f}")

    print(f"\n  Naked mean overlap:      {np.mean(n_overlaps):.3f}")
    print(f"  Compressed mean overlap: {np.mean(c_overlaps):.3f}")
    u, p = stats.mannwhitneyu(n_overlaps, c_overlaps, alternative="two-sided")
    print(f"  Mann-Whitney U={u:.0f}  p={p:.6f}  sig={'*' if p < 0.05 else 'ns'}")

    # ── 2e. Decision/fact growth rate analysis ───────────────────────────
    print("\n## 2e. State accumulation rates")
    for cond, data in [("naked", naked), ("compressed", compressed)]:
        decisions = [r["total_decisions"] for r in data]
        facts = [r["total_facts"] for r in data]
        tokens = [r["scaffold_tokens"] for r in data]

        # Linear regression: turn → token count (is it truly growing?)
        turns = np.arange(1, 21)
        slope_t, intercept_t, r_t, p_t, se_t = stats.linregress(turns, tokens)
        slope_d, _, r_d, p_d, _ = stats.linregress(turns, decisions)
        slope_f, _, r_f, p_f, _ = stats.linregress(turns, facts)

        print(f"\n  {cond}:")
        print(f"    Token growth:    {slope_t:+.1f} tokens/turn (R²={r_t**2:.3f}, p={p_t:.6f})")
        print(f"    Decision growth: {slope_d:+.2f} decisions/turn (R²={r_d**2:.3f}, p={p_d:.6f})")
        print(f"    Fact growth:     {slope_f:+.2f} facts/turn (R²={r_f**2:.3f}, p={p_f:.6f})")

    # ── 2f. Conditional entropy: does knowing the scaffold reduce ────────
    #         uncertainty about the response?
    print("\n## 2f. Conditional Entropy: H(response | scaffold)")
    for cond, data in [("naked", naked), ("compressed", compressed)]:
        s_bins = discretize([r["scaffold_entropy"] for r in data], bins=4)
        r_bins = discretize([shannon_entropy(r.get("response", "")) for r in data], bins=4)

        h_r = drv.entropy(r_bins)
        h_r_given_s = drv.entropy_conditional(r_bins, s_bins)

        print(f"  {cond:12s}: H(response) = {h_r:.4f}  H(response|scaffold) = {h_r_given_s:.4f}  reduction = {h_r - h_r_given_s:.4f} bits")

    # ── 2g. Transfer entropy (scaffold_t → response_t) ───────────────────
    print("\n## 2g. Transfer Entropy: scaffold(t-1) → response(t)")
    print("  (Does previous scaffold predict current response better than current scaffold alone?)")
    for cond, data in [("naked", naked), ("compressed", compressed)]:
        if len(data) < 3:
            continue
        # scaffold entropy at t-1, scaffold entropy at t, response entropy at t
        s_prev = discretize([r["scaffold_entropy"] for r in data[:-1]], bins=3)
        r_curr = discretize([shannon_entropy(r.get("response", "")) for r in data[1:]], bins=3)
        s_curr = discretize([r["scaffold_entropy"] for r in data[1:]], bins=3)

        # TE = H(r_t | s_t) - H(r_t | s_t, s_{t-1})
        h_r_given_s = drv.entropy_conditional(r_curr, s_curr)
        # Joint condition: combine s_curr and s_prev into single variable
        joint_cond = s_curr * 10 + s_prev  # simple encoding
        h_r_given_joint = drv.entropy_conditional(r_curr, joint_cond)
        te = h_r_given_s - h_r_given_joint

        print(f"  {cond:12s}: TE = {te:.4f} bits")
        if te > 0.1:
            print("    → Previous scaffold state carries information about current response beyond current scaffold")
        else:
            print("    → Previous scaffold state adds little beyond current scaffold (memoryless / Markov)")


if __name__ == "__main__":
    analyze_matrix()
    analyze_lpci()
