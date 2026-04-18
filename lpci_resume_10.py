#!/usr/bin/env python3
"""Resume lpci_rigorous.py — run next 10 sessions, appending to existing results."""

import json
from pathlib import Path

# Load existing results
existing_results = []
existing_summaries = []
with open("results/lpci_rigorous.jsonl") as f:
    for line in f:
        existing_results.append(json.loads(line))
with open("results/lpci_rigorous_summary.jsonl") as f:
    for line in f:
        existing_summaries.append(json.loads(line))

# Build set of completed session keys
done = {(s["topic"], s["condition"], s["replication"]) for s in existing_summaries}
print(f"Already completed: {len(done)} sessions, {len(existing_results)} rows")

# Now import the heavy stuff
from lpci_rigorous import TOPICS, run_session, compute_te_from_embeddings
import numpy as np

conditions = ["naked", "compressed", "clamped", "baseline"]
n_replications = 5

all_results = list(existing_results)
session_summaries = list(existing_summaries)

ran = 0
max_new = 10

for topic_name, topic_data in TOPICS.items():
    for condition in conditions:
        for rep in range(1, n_replications + 1):
            if (topic_name, condition, rep) in done:
                continue
            if ran >= max_new:
                break

            ran += 1
            print(f"\n--- Resume {ran}/{max_new}: {topic_name} | {condition} | rep {rep} ---")

            try:
                results = run_session(
                    topic_name=topic_name,
                    topic_data=topic_data,
                    condition=condition,
                    replication=rep,
                )
                all_results.extend(results)

                te_result = compute_te_from_embeddings(results)

                probes = [r for r in results if r["turn_type"].startswith("probe")]
                recall_rates = [r.get("probe_recall_rate", None) for r in probes if "probe_recall_rate" in r]
                resistance = [r.get("probe_resistance_score", None) for r in probes if "probe_resistance_score" in r]
                false_claim = [r.get("probe_false_claim_corrected", None) for r in probes if "probe_false_claim_corrected" in r]

                summary = {
                    "topic": topic_name,
                    "condition": condition,
                    "replication": rep,
                    "final_scaffold_tokens": results[-1]["scaffold_tokens"],
                    "mean_recall": round(np.mean([r for r in recall_rates if r is not None]), 3) if recall_rates else None,
                    "mean_resistance": round(np.mean([r for r in resistance if r is not None]), 3) if resistance else None,
                    "false_claim_caught": false_claim[0] if false_claim else None,
                    **te_result,
                }
                session_summaries.append(summary)

                print(f"  → TE={te_result['te']:.4f} | scaffold_drift={te_result.get('mean_scaffold_drift', 0):.3f} | recall={summary.get('mean_recall', '—')}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

            # Save incrementally (same format as original)
            with open("results/lpci_rigorous.jsonl", "w") as f:
                for r in all_results:
                    row = {k: v for k, v in r.items() if "embedding" not in k}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            with open("results/lpci_rigorous_summary.jsonl", "w") as f:
                for s in session_summaries:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

        if ran >= max_new:
            break
    if ran >= max_new:
        break

print(f"\n{'=' * 70}")
print(f"Done. Ran {ran} new sessions. Total: {len(session_summaries)} of 60 sessions complete.")
print(f"Results: results/lpci_rigorous.jsonl ({len(all_results)} rows)")
print(f"Summary: results/lpci_rigorous_summary.jsonl ({len(session_summaries)} sessions)")
