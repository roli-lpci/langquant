# LangQuant — LPCI PROVED: Stateless Models Hold State Through Language

**Linguistically Persistent Cognitive Interface (LPCI) — validated 2026-03-28**

A stateless LLM with zero conversation history maintained full conversational coherence across 20 turns using only a refreshing language scaffold as state. The model never saw any prior messages. The scaffold IS the memory.

`input is output is input is output`

## What Was Proved

An LLM (qwen3.5:9b) received **only** a structured scaffold + the current user message each turn. No conversation history. No RAG. No external memory. After every turn, a small model (qwen3.5:4b) extracted state changes and refreshed the scaffold within a fixed token budget.

**Results (20-turn A/B test):**
- Turn 4: Model correctly recalled all prior decisions (1.0 recall)
- Turn 8: Model rejected a contradiction with an earlier decision (resisted "switch to GPT-4" when scaffold contained "state extractor is qwen3.5:4b")
- Turn 12: Model listed decisions from turns 1-11 accurately
- Turn 16: Model recalled the first topic discussed and connected it to turn 15's discussion
- Turn 20: Model listed all decisions in order, caught a false claim ("we never decided on a state extractor" — corrected it)

**Both conditions (naked and compressed) maintained continuity.** The scaffold works as state.

## Why This Matters

Traditional LLM sessions: `[turn1][turn2]...[turnN][current]` — grows linearly, eventually chokes.

LPCI: `[scaffold: fixed K tokens, refreshed][current turn]` — never grows. Turn 30,000 is identical to turn 20 from the model's perspective.

The context window becomes a feature, not a limitation. You use all of it for dense state, not diluted history. Fixed budget, infinite session.

## Architecture

```
User message
     ↓
[Scaffold (K tokens)] + [Current message] → Main model (qwen3.5:9b) → Response
     ↓
[Response + Scaffold] → State extractor (qwen3.5:4b) → Delta (JSON)
     ↓
Apply delta → Refreshed scaffold (still K tokens) → Next turn
```

The model is a pure function. The scaffold is the program. The output of one pass feeds the compression of the next.

## Core Thesis

Formulated ~summer 2025 as LPCI (Linguistically Persistent Cognitive Interface):

- **Linguistically** — the medium is language, not tensors
- **Persistent** — survives across the stateless inference boundary
- **Cognitive** — does thinking-work (attention steering, probability reshaping), not just storage
- **Interface** — sits between sessions and the stateless model

## Prior Empirical Proof

LPCI builds on a lineage of experiments, all pointing to the same thesis:

- **TierJump** — scaffolded Haiku beats raw Sonnet on eval/content/research tasks (7 experiments, PF-001–PF-007)
- **scaffold-independence** — contrastive markers: +18-27pp accuracy on Banking77/CLINC150/MASSIVE
- **QuickThink** — compressed planning grammar (`g:;c:;s:;r:`) in 6-16 tokens, 30,900+ trials
- **Hypothesis Scaffold** — zero-LLM recursive engine: 113 hypotheses in 70 seconds from structured artifacts
- **Epistemic Experiments** — matched-pair probes proving linguistic framing steers model behavior

## Repo Contents

- `lpci.py` — Core LPCI prototype (SessionState, LPCISession, state extraction, scaffold refresh)
- `lpci_test.py` — A/B continuity test (naked vs compressed, 20 turns, probes at 4/8/12/16/20)
- `run_experiment.py` — LangQuant experiment harness (single-shot scaffold amplification measurement)
- `results/lpci_ab_test.jsonl` — **The proof.** 40 rows (20 naked + 20 compressed), full scaffold snapshots every turn, delta traces, probe evaluations.
- `results/full_run_v1.jsonl` — 575-trial matrix run (4 models x 5 conditions x 12 tasks x 3 runs)
- `LOG.md` — Complete project log with analysis and honest assessments
- `TODO.md` — Future work and practical applications

## What's Next

- Scaffold quantization: minimum viable scaffold size per model size
- Per-token ablation: which scaffold tokens carry the signal (semantic curvature)
- Signal-fingerprint methodology applied to scaffold compression (embedding centroid stability)
- Scale test: 100+ turns, 1000+ turns
- Cross-model: does a scaffold built by one model work when injected into another?

## Hermes Labs, 2026

*"Take a bunch of empty words and make them mean something."*
