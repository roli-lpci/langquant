#!/usr/bin/env python3
"""Post-process rigorous run: batch-embed scaffold+response, compute TE per session.

Run this AFTER lpci_rigorous.py finishes. It reads results/lpci_rigorous.jsonl,
embeds all scaffold_text and response fields, computes transfer entropy per session,
and updates results/lpci_rigorous_summary.jsonl with real TE values.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path

# Reuse helpers from lpci_rigorous
from lpci_rigorous import embed_text, cosine_similarity, compute_te_from_embeddings


def main():
    results_path = Path("results/lpci_rigorous.jsonl")
    summary_path = Path("results/lpci_rigorous_summary.jsonl")

    if not results_path.exists():
        print("ERROR: results/lpci_rigorous.jsonl not found. Run lpci_rigorous.py first.")
        sys.exit(1)

    # Load all results
    print("Loading results...")
    all_results = []
    with open(results_path) as f:
        for line in f:
            all_results.append(json.loads(line))
    print(f"  {len(all_results)} turns loaded")

    # Group by session
    sessions = {}
    for r in all_results:
        key = (r["topic"], r["condition"], r["replication"])
        if key not in sessions:
            sessions[key] = []
        sessions[key].append(r)

    print(f"  {len(sessions)} sessions found")

    # Batch embed all texts
    print(f"\nEmbedding {len(all_results) * 2} texts (scaffold + response per turn)...")
    t0 = time.monotonic()
    total = len(all_results)

    for i, r in enumerate(all_results):
        scaffold_text = r.get("scaffold_text", "")
        response_text = r.get("response", "")

        r["_scaffold_embedding"] = embed_text(scaffold_text[:2000]) if scaffold_text else []
        r["_response_embedding"] = embed_text(response_text[:2000]) if response_text else []

        if (i + 1) % 20 == 0 or i == total - 1:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    embed_time = time.monotonic() - t0
    print(f"  Done in {embed_time:.0f}s ({embed_time/60:.1f}min)")

    # Compute TE per session
    print(f"\nComputing transfer entropy for {len(sessions)} sessions...")
    session_te = {}
    for key, turns in sorted(sessions.items()):
        topic, condition, rep = key
        te_result = compute_te_from_embeddings(turns)
        session_te[key] = te_result
        print(f"  {topic:12s} | {condition:12s} | r{rep} | TE={te_result['te']:.4f} | drift={te_result.get('mean_scaffold_drift', 0):.3f}")

    # Update summary file
    print(f"\nUpdating {summary_path}...")
    summaries = []
    with open(summary_path) as f:
        for line in f:
            summaries.append(json.loads(line))

    for s in summaries:
        key = (s["topic"], s["condition"], s["replication"])
        if key in session_te:
            te = session_te[key]
            s["te"] = te["te"]
            s["mean_scaffold_response_cosine"] = te.get("mean_scaffold_response_cosine")
            s["mean_scaffold_drift"] = te.get("mean_scaffold_drift")
            s["mean_response_drift"] = te.get("mean_response_drift")

    with open(summary_path, "w") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  Updated {len(summaries)} session summaries with TE values")

    # Quick summary table
    print(f"\n{'='*70}")
    print("TE RESULTS BY CONDITION")
    print(f"{'='*70}")

    by_condition = {}
    for s in summaries:
        c = s["condition"]
        if c not in by_condition:
            by_condition[c] = []
        by_condition[c].append(s)

    print(f"\n{'Condition':>12s}  {'TE mean':>8s}  {'TE std':>7s}  {'S-drift':>8s}  {'Recall':>7s}")
    for c in ["naked", "compressed", "clamped", "baseline"]:
        data = by_condition.get(c, [])
        if not data:
            continue
        tes = [d["te"] for d in data if d.get("te") is not None]
        drifts = [d.get("mean_scaffold_drift", 0) for d in data]
        recalls = [d["mean_recall"] for d in data if d.get("mean_recall") is not None]

        if tes:
            print(f"{c:>12s}  {np.mean(tes):8.4f}  {np.std(tes):7.4f}  {np.mean(drifts):8.4f}  {np.mean(recalls):7.3f}")

    # Significance tests
    from scipy import stats as sp_stats
    print(f"\n## Mann-Whitney U: TE (naked vs each)")
    naked_te = [d["te"] for d in by_condition.get("naked", []) if d.get("te") is not None]
    for c in ["compressed", "clamped", "baseline"]:
        other_te = [d["te"] for d in by_condition.get(c, []) if d.get("te") is not None]
        if len(naked_te) >= 3 and len(other_te) >= 3:
            u, p = sp_stats.mannwhitneyu(naked_te, other_te, alternative="two-sided")
            print(f"  naked vs {c:12s}: U={u:.0f}  p={p:.4f}  sig={'*' if p < 0.05 else 'ns'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
