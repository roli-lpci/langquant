# LangQuant — Future Work & Practical Applications

## Active Experiments
- [ ] Full matrix run: 4 models × 5 conditions × 12 tasks × 3 runs (running as of 2026-03-28)
- [ ] Cross-architecture comparison: deepseek-r1:8b, gemma3:4b, mistral:7b (after main run)
- [ ] Add qwen3.5:27b to the matrix (run on dedicated GPU hardware)

## Core Research
- [ ] Validate behavioral complexity metric against human judgment (is our composite meaningful?)
- [ ] Per-token ablation study: drop scaffold tokens one at a time, measure performance delta (LLMLingua-style). Identifies which tokens carry the signal.
- [ ] Scaling law: does compression ratio follow a predictable curve across model sizes?
- [ ] PF-006 explanation: measure scaffold entropy vs model's existing capability on operational tasks. Hypothesis: high overlap = interference, measurable as redundancy.
- [ ] Formalize the core claim: "For model of size P, there exists language state S such that LLM(S) ≈ behavior of model kP on naive prompt."

## Practical Applications (Priority Order)

### APP-01: Memory Compression Layer (MAYBE — needs experiment first)
**What:** Compress retrieved memory facts into structured scaffolds before context injection.
**Open question:** Is this just summarization, or does compressed memory actually steer behavior better than raw facts? Need to test before building.
**Experiment idea:** Retrieve 20 facts raw vs compressed scaffold → measure response quality on same task. If no difference, it's summarization and not worth it. If compressed steers better (especially on small models with tight context), that's a real finding.
**Status:** Parked. Focus is on continuity (APP-02) first.

### APP-02: Rolling Session Scaffold (Linguistic Persistence Layer)
**What:** Fixed-budget language state (~500 tokens) that refreshes every N turns, replacing the growing context window paradigm. The model stays stateless; the scaffold IS the state.
**Why:** Turn 100 stays as coherent as turn 1. Small models don't choke on 8000 tokens of diluted history. Context window becomes a feature, not a limitation.
**Builds on:** Hypothesis Scaffold pattern (calibrate → artifacts → deterministic recursion). Early philosophical work on "linguistically persistent cognitive interface" (summer 2025 docs).
**Effort:** 1-2 days for prototype. This is potentially the core product.
**Architecture:**
  - After each turn: compress conversation into structured scaffold (could use small model or rules-based)
  - Before each turn: inject scaffold as prefix
  - Scaffold refreshes within fixed token budget — never grows, never truncates
  - Traditional: [turn1][turn2]...[turnN][current] — grows, dilutes
  - This: [scaffold: 500 tokens, refreshed][current turn] — fixed, dense

### APP-03: Router → Scaffold Auto-Injection in Agent Pipeline
**What:** Wire the existing centroid router (v7, live in production) to automatically select and inject the right scaffold per task type.
**Why:** Every agent request gets the optimal scaffold without manual selection.
**Builds on:** Centroid router v7 + TierJump scaffold type map (contrastive for classification, QuickThink for research, nothing for operational).
**Effort:** Half a day. The pieces exist — just need wiring.
**Note:** Scaffold router is already partially live in the agent pipeline.

### APP-04: Context Window Multiplier
**What:** Trade context window for scaffold density. 4k context with 500-token refreshing scaffold = functionally infinite session state.
**Why:** 2-3x effective context on small models. Makes 0.8b models viable for long sessions.
**Builds on:** LangQuant results (need full run data to calibrate compression ratios per model size).
**Effort:** Research project. Depends on APP-02 working first.

## Paper Ideas
- [ ] "Language as State: Formalizing Prompt Compression via Information-Theoretic Scaffolds" — the main LangQuant paper
- [ ] "When Scaffolding Looks Like Attacking" — T-022 anomaly from TierJump (Haiku's safety training classified prefix scaffold as prompt injection)
- [ ] "Linguistic Persistence: Fixed-Budget State Management for Stateless LLMs" — APP-02 as a paper

## Infrastructure
- [ ] Set up LangQuant as a public GitHub repo
- [ ] Add Inspect AI evals for behavioral complexity validation
- [ ] Build analysis/reporting script for LangQuant results
