# Changelog

## v0.0.8 (2026-03-28)

### Added
- Core LPCI prototype (`lpci.py`): SessionState, LPCISession, state extraction, scaffold refresh, interactive CLI
- A/B continuity test (`lpci_test.py`): 20 turns × 2 conditions, probes, scaffold evaluation, delta tracing
- Information-theoretic analysis (`analyze_results.py`): MI, KL divergence, transfer entropy via pyitlib + scipy
- Single-shot scaffold amplification harness (`run_experiment.py`): 4 models × 5 conditions × 12 tasks × 3 runs
- Full result datasets: 40-row LPCI proof + 619-trial matrix run
- README with complete methodology, results, architecture, and honest caveats

### Key Results
- Stateless LLM maintained full 20-turn coherence via refreshing language scaffold
- Transfer entropy ≈ 0.085 bits (compressed scaffold is Markov — complete state)
- 2.5x compression at turn 20 (789 tokens scaffold vs 1,945 tokens conversation)
