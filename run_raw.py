#!/usr/bin/env python3
"""Run ONLY the raw condition (15 sessions) and append to existing rigorous data."""

import json
import numpy as np
from pathlib import Path

from lpci_rigorous import (
    TOPICS, run_session, compute_te_from_embeddings,
)


def main():
    results_path = Path("results/lpci_rigorous.jsonl")
    summary_path = Path("results/lpci_rigorous_summary.jsonl")

    # Load existing data
    existing_results = []
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                existing_results.append(json.loads(line))
    print(f"Existing: {len(existing_results)} turn rows")

    existing_summaries = []
    if summary_path.exists():
        with open(summary_path) as f:
            for line in f:
                existing_summaries.append(json.loads(line))
    print(f"Existing: {len(existing_summaries)} session summaries")

    # Check what's already done
    done = {(s["topic"], s["condition"], s["replication"]) for s in existing_summaries}

    condition = "raw"
    n_replications = 5
    total_new = sum(
        1 for t in TOPICS for r in range(1, n_replications + 1)
        if (t, condition, r) not in done
    )
    print(f"\nNew sessions to run: {total_new}")

    if total_new == 0:
        print("All raw sessions already done.")
        return

    session_count = 0
    for topic_name, topic_data in TOPICS.items():
        for rep in range(1, n_replications + 1):
            if (topic_name, condition, rep) in done:
                print(f"  SKIP {topic_name}|{condition}|r{rep} (already done)")
                continue

            session_count += 1
            print(f"\n--- Session {session_count}/{total_new}: {topic_name} | {condition} | rep {rep} ---")

            try:
                results = run_session(
                    topic_name=topic_name,
                    topic_data=topic_data,
                    condition=condition,
                    replication=rep,
                )
                existing_results.extend(results)

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
                existing_summaries.append(summary)

                print(f"  -> recall={summary.get('mean_recall', '—')} | resistance={summary.get('mean_resistance', '—')} | tokens={summary['final_scaffold_tokens']}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

            # Save incrementally
            with open(results_path, "w") as f:
                for r in existing_results:
                    row = {k: v for k, v in r.items() if "embedding" not in k}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            with open(summary_path, "w") as f:
                for s in existing_summaries:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Quick summary
    raw_summaries = [s for s in existing_summaries if s["condition"] == "raw"]
    if raw_summaries:
        recalls = [s["mean_recall"] for s in raw_summaries if s.get("mean_recall") is not None]
        resists = [s["mean_resistance"] for s in raw_summaries if s.get("mean_resistance") is not None]
        tokens = [s["final_scaffold_tokens"] for s in raw_summaries]
        print(f"\n{'='*50}")
        print(f"RAW CONDITION RESULTS ({len(raw_summaries)} sessions)")
        print(f"  Recall:     {np.mean(recalls):.3f} +/- {np.std(recalls):.3f}" if recalls else "  Recall: N/A")
        print(f"  Resistance: {np.mean(resists):.3f} +/- {np.std(resists):.3f}" if resists else "  Resistance: N/A")
        print(f"  Tokens:     {np.mean(tokens):.0f}")
        print(f"{'='*50}")

    print(f"\nTotal data: {len(existing_results)} turns, {len(existing_summaries)} sessions")
    print("Done.")


if __name__ == "__main__":
    main()
