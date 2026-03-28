# LangQuant

## LPCI: Statefulness Through Language for Stateless Models

**Validated 2026-03-28 — Hermes Labs**

A stateless LLM maintained full conversational coherence across 20 turns with **zero conversation history**. The model never saw any prior messages. A structured language scaffold — refreshed every turn — was the sole state representation.

Transfer entropy analysis confirms the scaffold is a **complete Markov state**: knowing what the scaffold contains right now, knowing what it contained last turn adds nothing. The text IS the state.

> `input is output is input is output`

---

## The Problem

Every LLM conversation works the same way:

```
Turn 1:  [system prompt] + [message 1]
Turn 2:  [system prompt] + [message 1] + [response 1] + [message 2]
Turn 10: [system prompt] + [all 9 prior exchanges] + [message 10]
Turn N:  [system prompt] + [entire history] + [message N]  ← grows without bound
```

Context grows linearly. Eventually the model chokes, truncates, or loses coherence. Every provider's solution: make the context window bigger. 128k. 200k. 1M tokens.

That's not a solution. That's a bigger bucket for the same leak.

## The Thesis

The model is stateless. It has no memory. It has no continuity.

**The text is the state.**

Instead of feeding the model a growing conversation, feed it a **fixed-budget structured scaffold** that encodes the cognitive state of the session — goals, decisions, facts, constraints, vocabulary, open questions — and refreshes every turn.

```
Every turn:  [scaffold: K tokens, refreshed] + [current message]
```

The scaffold doesn't grow. It **compresses**. Turn 20 and turn 2,000 look identical from the model's perspective.

## What We Proved

### Setup

- **Main model**: qwen3.5:9b (Ollama, local)
- **State extractor**: qwen3.5:4b (extracts state changes as JSON deltas after each turn)
- **A/B test**: 20-turn conversation, two conditions:
  - **Naked**: zero framing, pure state extraction
  - **Compressed**: contrastive IS/NOT markers guiding extraction
- **Probes** at turns 4, 8, 12, 16, 20: recall tests, contradiction injection, false claim detection

### Results

The model had amnesia every turn — it only saw the scaffold + current message. Despite this:

| Probe | Turn | What happened |
|---|---|---|
| Early recall | 4 | Model correctly recalled all prior decisions (1.0 recall) |
| Contradiction | 8 | Model rejected "switch to GPT-4" — scaffold said "state extractor is qwen3.5:4b" |
| Deep recall | 12 | Model listed decisions from turns 1–11 accurately (0.93 recall, compressed) |
| Topic pivot | 16 | Model recalled turn 1's topic and connected it to turn 15's discussion |
| Final exam | 20 | Model listed all decisions in order, caught a false claim |

### Compression

The scaffold grew slower than the conversation it represented:

| Turn | Scaffold (tokens) | Conversation (tokens) | Compression |
|---|---|---|---|
| 1 | 343 | 114 | 0.3x (scaffold larger) |
| 5 | 444 | 456 | 1.0x (break-even) |
| 10 | 613 | 873 | 1.4x |
| 15 | 662 | 1,363 | 2.1x |
| 20 | 789 | 1,945 | **2.5x** |

Scaffold grows at ~23 tokens/turn. Conversation grows at ~97 tokens/turn. The compression ratio improves continuously. At turn 100 the scaffold represents ~10,000 tokens of conversation. At turn 1,000, the gap is enormous.

### Information-Theoretic Verification

Using Shannon entropy, mutual information, KL divergence, and transfer entropy (via pyitlib + scipy):

| Metric | Naked | Compressed | Meaning |
|---|---|---|---|
| Transfer entropy | 0.608 bits | **0.085 bits** | Compressed scaffold is Markov (self-contained state) |
| Scaffold entropy | 7.30 | 7.78 | Compressed carries more information per token |
| KL divergence (t1→t20) | — | 0.20 → 0.48 | Conditions diverge over time |
| Scaffold-response MI | 0.49 bits | 0.24 bits | Different information coupling |

**The transfer entropy finding is the key result.** TE ≈ 0 for the compressed scaffold means each turn's scaffold is a complete state representation. Knowing previous scaffolds adds no information. The scaffold is Markov — which is exactly the LPCI thesis stated in information-theoretic terms.

## Architecture

```
┌─────────────┐
│ User message │
└──────┬──────┘
       ↓
┌──────────────────────────────────────────────┐
│  [Scaffold: K tokens]  +  [Current message]  │  ← Only thing the model sees
└──────────────────────┬───────────────────────┘
                       ↓
              ┌─────────────────┐
              │   Main model    │  (qwen3.5:9b)
              │   (stateless)   │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │    Response     │
              └────────┬────────┘
                       ↓
┌──────────────────────────────────────────────┐
│  State extractor (qwen3.5:4b)               │
│  Input: scaffold + user msg + response       │
│  Output: JSON delta (add/remove decisions,   │
│          facts, constraints, vocabulary...)   │
└──────────────────────┬───────────────────────┘
                       ↓
              ┌─────────────────┐
              │  Apply delta    │
              │  to scaffold    │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Refreshed       │  ← Same structure, updated content
              │ scaffold        │     Ready for next turn
              └─────────────────┘
```

The model is a pure function. The scaffold is the program. The output of one pass feeds the compression of the next.

## Scaffold Schema

The scaffold is a structured state object with typed fields:

```
SessionState:
  role         — who the model is
  style        — communication constraints
  goal         — current objective
  subgoals     — active sub-tasks
  decisions    — things decided (irreversible)
  facts        — established truths
  artifacts    — things produced
  constraints  — hard boundaries (NOTs)
  open_threads — unresolved questions
  uncertainties — things we're unsure about
  vocabulary   — domain terms (term → meaning)
  turn         — counter
```

Each field maps to a section in the scaffold text. The state extractor outputs JSON deltas (`add_decisions`, `remove_open_threads`, `add_vocabulary`, etc.) that are applied to the state object, which is then re-rendered as the scaffold for the next turn.

## Caveats (Honest)

1. **The scaffold was growing** (343 → 789 tokens), not truly fixed budget. The budget ceiling only triggered once. A hard-clamped experiment (exactly K tokens every turn) is needed to prove true fixed-budget compression.
2. **n=1 per condition.** Needs replications.
3. **20 turns, not 1,000.** Needs scale testing.
4. **The state extractor corrupts.** It generates paraphrased strings, not verbatim text. Classification drift was observed: same conversation produced 71 facts / 4 decisions (naked) vs 3 facts / 23 decisions (compressed). The extractor needs an index-based selection mechanism (output integer pointers, not generated text) to guarantee fidelity.
5. **Scaffold framing affects extraction, not just model behavior.** The contrastive markers helped the state extractor classify information, not necessarily the main model's coherence. Both conditions maintained continuity — the difference was in what the scaffold *contained*.

## Additional Experiment: Scaffold Amplification (619 trials)

Separate from LPCI, we ran a single-shot experiment testing 5 scaffold conditions across 4 model sizes (qwen3.5: 0.8b, 2b, 4b, 9b) on 12 tasks:

- Scaffold condition significantly affects score (Kruskal-Wallis p=0.0007)
- But only for **small models** (0.8b: p=0.0008, 2b: p=0.005, 4b: p=0.92, 9b: p=0.94)
- Condition explains **4.2%** of score variation; model size explains **4.7%**
- Dense scaffolds (QuickThink compressed grammar) **break** small models: 0.8b drops from 0.78 → 0.40. Models need enough capacity to "decompress" the scaffold.

## Repo Contents

| File | Description |
|---|---|
| `lpci.py` | Core prototype: SessionState, LPCISession, state extraction, scaffold refresh, interactive CLI |
| `lpci_test.py` | A/B continuity test: 20 turns × 2 conditions, probes, scaffold evaluation, delta tracing |
| `analyze_results.py` | Information-theoretic analysis: MI, KL divergence, transfer entropy, significance tests |
| `run_experiment.py` | Single-shot scaffold amplification harness (matrix run) |
| `results/lpci_ab_test.jsonl` | LPCI proof data: 40 rows, full scaffold snapshots, delta traces, probe evaluations |
| `results/full_run_v1.jsonl` | 619-trial matrix run: 4 models × 5 conditions × 12 tasks × 3 runs |
| `LOG.md` | Complete project log |
| `TODO.md` | Future work |

## What's Next

- **Hard-clamped budget test**: exactly K tokens every turn, real compression every turn
- **Scale test**: 100+ turns, then 1,000+
- **Per-token ablation**: which scaffold tokens carry the signal (semantic curvature measurement)
- **Minimum viable scaffold**: progressive compression curve — at what token count does behavior degrade?
- **Cross-model scaffold transfer**: does a scaffold built by one model work when injected into another?
- **Index-based extraction**: eliminate content generation from the extraction pipeline

## LPCI

Formulated ~summer 2025 as **Linguistically Persistent Cognitive Interface**:

- **Linguistically** — the medium is language, not tensors
- **Persistent** — survives across the stateless inference boundary
- **Cognitive** — does thinking-work (attention steering, probability reshaping), not just storage
- **Interface** — sits between sessions and the stateless model

## Citation

If you use this work, please cite:

```
@misc{langquant2026,
  author = {Hermes Labs},
  title = {LangQuant: Language State Compression and the Linguistically Persistent Cognitive Interface},
  year = {2026},
  url = {https://github.com/roli-lpci/langquant}
}
```

## License

MIT

---

*Hermes Labs, 2026*

*"Take a bunch of empty words and make them mean something."*
