# AGENTS.md — Agent Instructions for LangQuant

## What this project is

LangQuant is a research prototype proving the LPCI (Linguistically Persistent Cognitive Interface) thesis: a stateless LLM can maintain full conversational coherence using only a refreshing structured language scaffold — no conversation history.

## Architecture

```
langquant/
├── lpci.py              # Core: SessionState, LPCISession, extraction, scaffold refresh, CLI
├── lpci_test.py         # A/B continuity test (20 turns × 2 conditions)
├── analyze_results.py   # Information-theoretic analysis (MI, KL, transfer entropy)
├── run_experiment.py    # Scaffold amplification matrix harness
├── results/             # JSONL data files (proof + matrix run)
├── tasks/               # Task definitions for matrix run
├── LOG.md               # Development log
└── TODO.md              # Future work
```

## Key concepts

1. **SessionState**: Typed dataclass with 12 fields (role, style, goal, subgoals, decisions, facts, artifacts, constraints, open_threads, uncertainties, vocabulary, turn). This is the scaffold.
2. **State extractor**: Smaller model (qwen3.5:4b) that reads scaffold + message + response and outputs JSON deltas (add/remove operations per field).
3. **Scaffold refresh**: Apply deltas to SessionState, re-render as text, inject as sole context for next turn.
4. **Transfer entropy ≈ 0**: The compressed scaffold is Markov — knowing previous scaffolds adds no information beyond the current one.

## Running experiments

Requires [Ollama](https://ollama.ai) with models pulled locally:
```bash
ollama pull qwen3.5:9b
ollama pull qwen3.5:4b
```

Run the LPCI A/B test:
```bash
python lpci_test.py
```

Run the scaffold amplification matrix:
```bash
python run_experiment.py
```

Analyze results:
```bash
python analyze_results.py
```

## Running tests

```bash
pip install pytest pyitlib scipy numpy
pytest -v
```

## Style

- Pure Python, minimal dependencies
- Results stored as JSONL for reproducibility
- Honest about limitations (see Caveats in README)
- Google-style docstrings
