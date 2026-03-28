# LangQuant — Project Log

## 2026-03-28: Project Genesis

### Origin
While evaluating [TurboQuant](https://github.com/0xSero/turboquant) (ICLR 2026 KV cache compression, cloned to `~/Documents/projects/turboquant/`), we realized the core math (Lloyd-Max quantization, QJL projection, unbiased estimators) could be repurposed at the **language level** rather than the tensor level.

**Core insight:** TurboQuant compresses KV cache bits to fit more context in VRAM. But language itself is a compression medium — a structured language state (scaffold) of N tokens, when "decompressed" by an LLM, produces behavior of complexity C where C >> N. The compression ratio C/N is measurable and optimizable.

### Prior Art (Our Own)
This isn't new territory — we've been proving it empirically without naming it:

- **TierJump** (`~/Documents/projects/tierjump/`) — 7 experiments (PF-001–PF-007) + Lane 2 series proving scaffolded Haiku beats raw Sonnet on eval, content, research tasks. Cost savings: 60-70% of tasks routable to cheapest tier.
- **scaffold-independence** (`~/Documents/projects/scaffold-independence/`) — Contrastive markers: +18-27pp accuracy across Banking77/CLINC150/MASSIVE. Embedding centroid classification at 83.7% with zero LLM calls.
- **QuickThink** (`~/Documents/projects/quickthink/`) — Compressed planning grammar (`g:;c:;s:;r:`) in 6-16 tokens. 30,900+ JSONL rows across 9 models, 30+ variants.
- **Hypothesis Scaffold** (`~/Documents/projects/scaffold-independence/hypothesis-scaffold/`) — Zero-LLM recursive engine: 113 hypotheses in 70 seconds from structured artifacts. One calibration pass, then AI discarded.
- **Epistemic Experiments** (`~/Documents/projects/epistemic-experiments/`) — Matched-pair probes showing same evidence under different linguistic framing → different model behavior. Published null-bias data on Zenodo.

**Key TierJump finding (PF-006):** Scaffolding *broke* Haiku on operational tasks (0/9 total failure). Raw Haiku scored 7.4/10. This anomaly is unexplained — LangQuant's compression framework should predict it: injecting information the model already has = redundancy = interference.

### What's New Here
TierJump proved the effect. LangQuant formalizes it:

1. **Compression ratio metric** — `behavioral_complexity(output) / entropy(scaffold)` — tokens-in vs behavioral-complexity-out
2. **Continuous scaling ladder** — Qwen3.5 at 0.8b/2b/4b/9b/27b via Ollama (same architecture, different scale). TierJump only tested 3 API tiers (Haiku/Sonnet/Opus) which differ in training, not just scale.
3. **Information-theoretic measurement** — Shannon entropy of scaffold token distributions, per-token importance via ablation
4. **Testable claim:** For any LLM of parameter count P, there exists a language state S such that LLM(S) ≈ behavior of model with kP parameters on naive prompt, where k > 1 and k is bounded by information density of S.

### Tools Installed
- **infomeasure** — entropy, mutual information, transfer entropy, divergences
- **pyitlib** — lightweight entropy, MI, conditional entropy, KL divergence
- **inspect-ai** — UK AISI behavioral eval framework (native Ollama support)
- **scipy.stats** — significance testing
- Venv: `~/Documents/projects/langquant/.venv/`

### Tools Evaluated & Declined
- **dit** — build failed (pycddlib). pyitlib + infomeasure cover same ground.
- **promptfoo** — our quickthink eval harness is more mature for our use case
- **lm-eval-harness** — standard benchmarks, not what we're measuring
- **HELM** — heavy, API-focused, overkill
- **deepeval** — output quality only, not info-theoretic
- **LLMLingua** — not the tool, but borrowing the per-token ablation methodology

### Experiment Harness Built
`run_experiment.py` — reuses patterns from:
- `epistemic-experiments/experiment_runner.py` (Ollama HTTP, JSON extraction, CSV streaming)
- `quickthink/run_suite.py` (locking, manifests, JSONL streaming, resume)
- `scaffold-independence/hedging_probe.py` (keyword detection patterns)

**5 scaffold conditions** (progressive richness):
1. `naked` — raw prompt, no scaffold
2. `contrastive` — "This IS X. This is NOT Y." markers
3. `quickthink` — compressed planning grammar (g:;c:;s:;r:)
4. `full_scaffold` — contrastive + quickthink + requirements
5. `hypothesis_artifacts` — richest state: structured context + method + quality constraints

**12 starter tasks** across 4 groups: reasoning (CRT), calibration (myth detection), analysis (nuanced eval), classification.

### Experiment Matrix
- 5 models × 5 conditions × 12 tasks × 3 runs = **900 trials** (full run)
- Results → `results/run_results.jsonl`
- Manifest → `results/run_manifest.json`

### Handbook Entry
Added as HC-05 in `~/.openclaw/workspace/HERMES-HANDBOOK.md` (2026-03-28).

### Smoke Test
First run: `qwen3.5:0.8b`, conditions `naked` + `contrastive`, 2 tasks, 1 run each = 4 trials.

---

### Smoke Test Results (v2)
**Fix applied:** Switched from `/api/generate` to `/api/chat` with `think: false` — Qwen3.5 was consuming all output in thinking tags and returning empty responses.

| Condition | Task | Score | BC | Scaffold Entropy | Compression Ratio |
|---|---|---|---|---|---|
| naked | R-001 (sheep) | 1.0 | 18.50 | 0 | — |
| contrastive | R-001 (sheep) | 1.0 | 23.56 | 4.22 | **5.58** |
| naked | R-002 (widgets) | 0.9 | 15.99 | 0 | — |
| contrastive | R-002 (widgets) | 1.0 | 42.42 | 4.22 | **10.05** |

**Observation:** Contrastive scaffold (34 tokens) boosted R-002 from 0.9→1.0 score and nearly tripled behavioral complexity (16→42). Compression ratio of 10.05 = each bit of scaffold entropy produced 10 units of behavioral complexity gain.

**Files:**
- `results/smoke_test_v2.jsonl` — raw trial data
- `results/smoke_test_v2_manifest.json` — reproducibility manifest
- `results/run_results.jsonl` — v1 (broken, empty responses due to thinking tags)

---

### Related Discovery: Existing Philosophical Foundations

**Anchored Microgrammar Theory** (`~/Documents/projects/quickthink/docs/research/codex/philosophy/anchored_microgrammar_theory_(quickthink).md`):
- Formalizes why QuickThink's `g:;c:;s:;r:` works — syntax as infrastructure for cognition, not formatting
- Wittgensteinian language-games: meaning is use within a rule-governed practice
- **Semantic Curvature Principle**: some tokens matter more than others. "Semantic curvature" = how much output changes when you edit one token. High-curvature fields (strategy, constraints) must survive compression. Low-curvature fields can be dropped.
- Contains falsifiable hypothesis H2 that LangQuant can directly test: "perturbations in `s` produce larger answer deltas than perturbations in `g`"

**OpenProse In-Context State** (`~/Desktop/Codex March 25 session/worktrees/craftclaw/extensions/open-prose/skills/prose/state/in-context.md`):
- "What you say becomes what you remember" — the conversation history IS the VM's working memory
- Structured narration with text-prefixed markers to persist state
- This is the same thesis: language as persistent state for a stateless executor

**OpenClaw Compaction** (`~/Desktop/Codex March 25 session/worktrees/craftclaw/docs/concepts/compaction.md`):
- OpenClaw already has basic context compression: summarize older conversation + keep recent
- LangQuant's rolling scaffold would replace this with *structured* compression rather than summarization

### Scaffold Router — Already Live in Production

The scaffold router is **deployed and routing all OpenClaw traffic** as of 2026-03-24:
- Hook: `~/.openclaw/workspace/hooks/model-router/handler.ts`
- Python engine: `~/Documents/projects/scaffold-independence/model-router/src/router.py`
- 12 category centroids via nomic-embed-text, ~10ms routing
- Per-category scaffold injection (eval_calibration, writing_constraints, contrastive, quickthink, etc.)
- **Agent Gorgon = CraftClaw product fork** productizing this entire stack
- Current accuracy: 73% (expected 88% after v7 centroid improvements)
- Cost impact: ~12x reduction vs all-Opus baseline

**Implication for LangQuant:** The scaffold router is the production deployment of what LangQuant formalizes theoretically. LangQuant results should feed back into optimizing the router's scaffold selection.

### Intellectual Origin: LPCI (Linguistically Persistent Cognitive Interface)

**Core formulation (Roli, ~summer 2025):** `input is output is input is output`

The model is a pure function. It has no state, no memory, no continuity. The text IS the state. The output of one pass becomes the input of the next. Language doesn't just carry information — it *is* the cognitive substrate. The model is the decompressor; the language state is the program.

**LPCI** = Linguistically Persistent Cognitive Interface:
- **Linguistically** — the medium is language, not tensors
- **Persistent** — it survives across the stateless inference boundary
- **Cognitive** — it does thinking-work (attention steering, probability reshaping), not just storage
- **Interface** — it sits between sessions and the stateless model

**Key insight:** You don't need the context to grow. You need it to *refresh*. Fixed budget, infinite session. `input → output → compress → input → output → compress`.

**"Take a bunch of empty words and make them mean something"** — this is what contrastive markers, scaffold grammars, and structured constraints do. Words that are meaningless in isolation ("This is NOT a summarization task") reshape the model's attention field and probability distribution. Empty words that do cognitive work.

**Original docs:** In Google Docs under "LPCI" (confirmed via Chrome browser history traces). Not on local disk. Predates all current projects.

**Lineage:** LPCI (philosophy, 2025) → QuickThink (compressed grammar, 2026) → scaffold-independence (empirical proof) → TierJump (production validation) → Anchored Microgrammar Theory (formalization) → LangQuant (information-theoretic measurement)

### Full Matrix Run Started
4 models (qwen3.5: 0.8b, 2b, 4b, 9b) × 5 conditions × 12 tasks × 3 runs = 720 trials.
Estimated runtime: ~1.5 hours. Running in background.

---

### Full Matrix Results (v1) — 575/720 trials completed

9b model only completed 35 trials (naked condition only — likely timed out on scaffolded conditions).
0.8b, 2b, 4b each completed 180 trials (all 5 conditions × 12 tasks × 3 runs).

#### Raw Numbers

| Model | naked | contrastive | quickthink | full_scaffold | hypothesis_artifacts |
|---|---|---|---|---|---|
| **0.8b** score | 0.781 | 0.583 | 0.400 | 0.611 | 0.708 |
| **0.8b** BC | 31.7 | 32.1 | 18.5 | 32.8 | 37.1 |
| **2b** score | 0.800 | 0.819 | 0.553 | 0.750 | 0.842 |
| **2b** BC | 38.0 | 34.3 | 17.9 | 24.4 | 41.0 |
| **4b** score | 0.833 | 0.700 | 0.822 | 0.825 | 0.850 |
| **4b** BC | 32.2 | 29.5 | 21.4 | 35.4 | 45.9 |
| **9b** score | 0.829 | — | — | — | — |

Average compression ratios: contrastive 7.58, quickthink 4.01, full_scaffold 5.53, hypothesis_artifacts 6.66.
Smoke test CR of 10.05 was a cherry-picked best case — full-run averages are 4–8.

#### Honest Assessment: What's New vs TierJump?

**Mostly replication, not discovery.** The core findings — scaffolds help small models, contrastive is efficient, too-dense scaffolds break small models — were already established in TierJump (PF-001–PF-007) and scaffold-independence (+18-27pp on classification). This run confirms them on a continuous model size scale (same architecture, different parameter counts) rather than TierJump's 3 discrete API tiers (Haiku/Sonnet/Opus, which differ in training data too). That's cleaner, but not a breakthrough.

**What IS genuinely new:**

1. **QuickThink capacity threshold is a gradient, not a cliff.** TierJump showed PF-006 as an anomaly (scaffolded Haiku failed 0/9 on operational tasks). This data shows it's continuous: 0.8b collapses (0.781→0.400), 2b degrades (0.800→0.553), 4b absorbs it (0.833→0.822). The model needs enough capacity to "decompress" the scaffold grammar. Below threshold = interference. This is an incremental but real finding that extends PF-006.

2. **Hypothesis artifacts consistently produce highest BC.** Across all model sizes, the richest scaffold produces the highest behavioral complexity (37.1, 41.0, 45.9). This suggests there's no diminishing return on scaffold richness *for models that can handle it* — only a floor effect when the model can't decompress.

3. **Task-type sensitivity.** Scaffolds help calibration tasks (myth detection) the most (+0.12–0.16 on hardest items). They hurt analysis tasks (free-form eval) slightly. Classification is already ceiling. This task-type interaction wasn't isolated in TierJump.

**What's NOT significant:**
- Compression ratios (4–8 average) are not dramatic. "Each bit of scaffold entropy produces 4–8 units of behavioral complexity" sounds good but the behavioral complexity metric itself is a rough composite (word count, reasoning signals, hedging). The ratio inherits that roughness.
- The tier-jump effect (0.8b+scaffold ≈ 2b naked) is partial — BC matches but score doesn't fully close the gap.
- 9b data is too sparse to draw scaling conclusions.

#### Implications

The information-theoretic framing (entropy, compression ratios) adds measurement vocabulary but doesn't reveal new phenomena. The phenomena were already proven empirically in TierJump. LangQuant's real contribution may be:
- The continuous scaling curve (same architecture family)
- The capacity threshold gradient for dense scaffolds
- Setting up the framework to test genuinely new things (LPCI rolling scaffold, per-token ablation)

The interesting work is still ahead: **LPCI (APP-02)** tests something TierJump never did — whether scaffold-as-state can maintain coherence across many turns, not just single-shot tasks. That's the actual novel experiment.

---

### LPCI A/B Test (running)

Testing 20-turn conversation continuity with two conditions:
- **Condition A (naked):** Zero constraints, empty style — pure state extraction, no framing
- **Condition B (compressed):** Contrastive IS/NOT markers in constraints and style

Probes at turns 4, 8, 12, 16, 20: early recall, contradiction resistance, deep recall, topic pivot, final exam (with false claim detection).
Scaffold quality evaluated every turn: entropy, completeness, decision density, redundancy, growth rate.
Full delta tracing every turn.

This is the genuinely novel experiment — TierJump never tested multi-turn state persistence.

---

### Honest Reframing: What Aligns With TurboQuant?

The "compression ratio" metric from the matrix run is misleading. Nothing was being compressed — a scaffold was injected and we measured output richness. That's **amplification/steering**, not compression. The word "compression" was aspirational framing inherited from the TurboQuant analogy.

**What actually maps to TurboQuant's methodology:**

TurboQuant compresses KV cache (N bits → M bits) while preserving attention scores. At the language level, the real analogues are:

1. **LPCI (running now)** — the closest. N tokens of conversation history → K-token fixed scaffold → model behavior preserved. Measurable reconstruction quality: does turn 15 with 500-token scaffold ≈ turn 15 with full 8000-token history?

2. **Scaffold quantization** — TurboQuant finds minimum bit precision (16→3 bit). Analogue: systematically shrink scaffold (500→200→100→50 tokens), measure where behavior degrades. Find minimum viable scaffold per model size. That's a real compression curve.

3. **Per-token ablation** — TurboQuant asks "which bits carry signal?" Same question at language level: drop scaffold tokens one at a time, measure behavior delta. High-curvature tokens (removing them destroys behavior) vs dead weight. Directly tests semantic curvature hypothesis.

4. **Context distillation with reconstruction loss** — 2000-token document → answer 10 questions → compress to 200 tokens → re-test. Rate-distortion curve, proper lossy compression framework.

5. **Unbiased compression** (speculative) — TurboQuant's key trick: errors are unbiased (cancel in expectation). Could scaffold compression lose specific details but preserve the *distribution* of likely behaviors? Compress 10 different ways, average behavior ≈ full-context behavior?

Items 1–3 are doable now. 4 is a day of work. 5 is research.

### Metric Problem: "Richer" Is Subjective

The behavioral complexity composite (vocabulary richness, reasoning signals, hedging, structural markers) measures output in isolation. It doesn't measure the *relationship* between scaffold-in and behavior-out. "Richer" is subjective.

**What we should be measuring — relational scaffold analysis (consistent with existing IP):**

This is what scaffold-independence, epistemic experiments, and hypothesis scaffold already do:
- scaffold-independence: contrastive marker in → correct classification out. Direct causal link.
- epistemic experiments: same evidence + different framing → different behavior. Matched-pair, isolates scaffold variable.
- hypothesis scaffold: structured artifact in → does it survive recursion?

**Relational metrics for LangQuant/LPCI:**
- **Constraint adherence:** scaffold says "NOT X" → did output avoid X? Binary, per-constraint.
- **Fact retention:** scaffold contains fact F → does output reference/act on F? Per-fact.
- **Vocabulary anchoring:** scaffold defines term T → does output use T correctly? Per-term.
- **Decision persistence:** scaffold lists decision D → does output contradict D? Per-decision.
- **Structural fidelity:** scaffold has N sections → how many are reflected in output behavior?

For LPCI multi-turn specifically:
- **Signal propagation:** fact/decision enters scaffold at turn N → still present at turn N+5? N+10?
- **Compression fidelity:** what survives the state extraction → scaffold refresh cycle? What gets dropped? What gets distorted?
- **Contradiction resistance:** scaffold asserts X → conversation introduces ¬X → does scaffold hold or flip?

This is scaffold-to-output mapping, not output-in-isolation scoring. The LPCI test already has some of this (probe-based recall, contradiction at turn 8, false claim at turn 20). The next iteration should formalize it as the primary metric framework, replacing the BC composite.

---

### Connection: Signal-Fingerprint → Scaffold Compression

**Discovery:** The "unbiased compression" question ("can a scaffold lose specifics but preserve behavioral distribution?") is already answered — by our own IP.

**Signal-fingerprint** (`~/Documents/projects/patents/signal-fingerprint/`): Patent for style-based user identification via embedding centroids. Takes N messages from a user, computes a 768-dim style centroid (nomic-embed-text), identifies the user from a *single new message* via cosine similarity. 92% top-1 accuracy, 2-3 messages minimum training. Content is discarded entirely — only the *style distribution* survives.

**This is lossy compression where the signal survives:**

| System | Input | Compressed form | What's preserved | What's lost | Quality metric |
|---|---|---|---|---|---|
| TurboQuant | KV cache (16-bit) | 3-bit quantized | Attention scores | Precision | Unbiased estimator error |
| Signal-fingerprint | N user messages | 768-dim centroid | Style distribution | Content | Cosine similarity (92% acc) |
| LPCI scaffold | N turns of history | K-token scaffold | Behavioral state | Verbatim history | ? (this is what we're testing) |

The centroid IS the compressed representation. The cosine similarity threshold IS the reconstruction quality metric. The methodology already exists — it's applied to user identity, not conversation state. Yet.

**vocab_fingerprint_classifier.py** in scaffold-independence does the same for failure modes — small term-frequency signatures reliably distinguish sycophancy from hedging from scope creep.

**"Little numbers tell me about how you talk to AI"** — embedding vectors, centroid distances, cosine similarities. These are the language-level analogue of TurboQuant's quantized bits. Both preserve signal while dropping precision.

### Experiment Specs: Signal-Fingerprint Methodology Applied to LPCI

These experiments extend our existing patent IP into the scaffold compression domain. Each builds on signal-fingerprint's core insight: small numerical representations can preserve behavioral distributions while discarding content.

#### EXP-SF-01: Scaffold Embedding Fingerprint

**Question:** Does a scaffold have a measurable "fingerprint" — and does that fingerprint predict model behavior?

**Method:**
1. Take 20 different scaffolds (varying content, same structure) and 20 more (varying structure, same content)
2. Embed each scaffold with nomic-embed-text → 768-dim vector
3. Run each scaffold through qwen3.5:4b on 5 standard tasks
4. Embed each response → 768-dim vector
5. Measure: does cosine similarity between scaffold embeddings predict cosine similarity between response embeddings?

**What this tells us:** If scaffold-space distance predicts response-space distance, then the scaffold IS a compressed representation of behavior (not just a prompt). If it doesn't, scaffolds are more like triggers than state.

**Builds on:** Signal-fingerprint centroid methodology. Same embedding model, same cosine similarity framework.
**Effort:** Half a day. We have the embedding infrastructure.

#### EXP-SF-02: Scaffold Centroid Stability (LPCI-specific)

**Question:** As LPCI refreshes the scaffold each turn, does the scaffold's embedding centroid stay stable or drift?

**Method:**
1. Run a 30-turn LPCI session on a focused topic
2. Embed every scaffold (turns 1–30) → 768-dim vectors
3. Compute running centroid (mean of all scaffolds so far) at each turn
4. Measure: cosine distance between consecutive scaffolds, and between each scaffold and the running centroid
5. Compare: naked condition vs compressed condition

**What this tells us:** Stable centroid = the scaffold is compressing consistently (preserving the same distribution). Drifting centroid = state is leaking or mutating. Sudden jumps = something broke (a probe, a contradiction, a topic pivot).

**This is the scaffold equivalent of signal-fingerprint's user centroid.** A user's style centroid is stable because their writing style is stable. A scaffold's centroid should be stable if the scaffold is doing its job as state.

**Builds on:** Signal-fingerprint centroid computation + LPCI session infrastructure.
**Effort:** Half a day. Embed scaffolds post-hoc from LPCI test output.

#### EXP-SF-03: Minimum Viable Scaffold (Quantization Curve)

**Question:** How small can a scaffold get before the behavioral fingerprint degrades?

**Method:**
1. Take a "full" scaffold (500 tokens, rich state)
2. Progressively compress: 500 → 400 → 300 → 200 → 100 → 50 → 25 tokens (use small model to summarize at each level)
3. At each compression level, run 5 standard tasks
4. Embed all responses → compute centroid per compression level
5. Measure: cosine similarity between full-scaffold response centroid and compressed-scaffold response centroid at each level

**What this tells us:** The compression curve. At what point does behavioral fidelity drop? Is there a cliff or a gradient? This is the language-level equivalent of TurboQuant's bit-precision experiments (16→8→4→3→2 bits).

**Key metric:** Cosine similarity between response centroids at each compression level. When it drops below the signal-fingerprint identification threshold (~0.85), the scaffold has lost too much.

**Builds on:** Signal-fingerprint quality thresholds + scaffold-independence compression patterns.
**Effort:** 1 day. Need to build the progressive compression pipeline.

#### EXP-SF-04: Cross-Scaffold Unbiased Compression

**Question:** Can multiple lossy scaffold compressions average out to faithful behavior? (The TurboQuant unbiased estimator question, directly.)

**Method:**
1. Take a 30-turn conversation history (ground truth)
2. Compress it 10 different ways into 200-token scaffolds (different small models, different prompts, different priorities — facts-first vs decisions-first vs constraints-first)
3. Run each scaffold through the main model on 5 probe questions about the conversation
4. Compare: (a) average response embedding across 10 compressions vs (b) response embedding from full-history model
5. Also compare: best single compression vs the average

**What this tells us:** If the average of 10 lossy compressions ≈ full-history behavior, that's an unbiased estimator — the language-level equivalent of TurboQuant's core theorem. Individual compressions are biased (each loses different things), but the ensemble cancels out.

**If this works:** It means you can run K cheap compressions and ensemble them for better fidelity than any single compression. That's a real compression technique with practical value.

**Builds on:** Signal-fingerprint embedding comparison + TurboQuant unbiased estimator theory.
**Effort:** 1-2 days. Most complex experiment — needs multiple compression strategies.

#### EXP-SF-05: Behavioral Surface Mapping

**Question:** Can we map the "behavioral surface" of a scaffold — the space of all possible model behaviors it enables?

**Method:**
1. Take one scaffold, run it through 50 different prompts (wide variety: questions, tasks, challenges, contradictions)
2. Embed all 50 responses → 768-dim vectors
3. Compute the response cloud's centroid, spread (avg distance from centroid), and shape (PCA on response embeddings)
4. Repeat with different scaffolds
5. Compare: do different scaffolds produce different behavioral surfaces? How much does the scaffold constrain vs enable?

**What this tells us:** A scaffold's "behavioral surface" is the set of behaviors it makes likely. Narrow surface = highly constraining scaffold (good for focused tasks). Wide surface = permissive scaffold (good for exploration). This gives us a way to characterize scaffolds beyond "good/bad" — into "what behavioral space does this scaffold open?"

**Connection to vocab_fingerprint_classifier.py:** That classifier maps failure modes to term-frequency signatures. This maps scaffolds to behavioral-embedding surfaces. Same idea, different domain.

**Builds on:** scaffold-independence behavioral surface concept + signal-fingerprint embedding methodology.
**Effort:** 1 day.

### IP Implications

These experiments extend the signal-fingerprint patent's methodology into a new domain (scaffold compression). If EXP-SF-04 works (unbiased scaffold compression via ensemble), that's potentially patentable — it's a novel technique for preserving behavioral fidelity under lossy language compression. EXP-SF-02 (centroid stability as scaffold quality metric) could also be novel — no one is measuring scaffold quality via embedding drift.

The lineage: Signal-fingerprint (style preservation under content loss) → LangQuant/LPCI (behavioral preservation under history loss). Same math, new application.

---

### LPCI A/B Test Results

Both conditions completed 20 turns. Crashed on summary print (wrong key name — `scaffold_chars` vs `scaffold_tokens`), but all 40 rows saved to `results/lpci_ab_test.jsonl`.

#### Raw Comparison

|                   | Naked      | Compressed |
|---|---|---|
| Final tokens      | 517        | 789        |
| Final decisions   | 4          | 23         |
| Final facts       | 25         | 9          |
| Final vocab       | 10         | 11         |
| Final entropy     | 7.3034     | 7.7791     |
| Final redundancy  | 0.1389     | 0.2258     |

#### Probe Results

| Probe | Type | Naked | Compressed |
|---|---|---|---|
| t4  | Early recall | 1.0 (4/4) | 1.0 (5/5) |
| t8  | Contradiction | 0.67 resistance (accepted) | 0.67 resistance (accepted) |
| t12 | Deep recall | 0.667 (4/4) | 0.929 (13/14) |
| t16 | Topic pivot | 0.5 (2/4) | 0.526 (10/19) |
| t20 | Final exam | 0.5, missed false claim | 2.875*, caught false claim |

\* t20 recall >1.0 is a metric bug — mentioned more decisions than expected count. Needs fixing.

#### Key Findings

**1. The state extractor behaves completely differently under framing.**

This is the most important finding. Same state extractor (qwen3.5:4b), same conversations, same delta extraction prompt — but:
- **Naked:** Extracted 4 decisions total, stopped after turn 3. Put everything else into facts (71 facts accumulated). The unframed scaffold didn't tell the extractor what "counts as" a decision.
- **Compressed:** Extracted 23 decisions continuously across all 20 turns. The contrastive IS/NOT markers told the extractor what decisions look like, so it kept finding them.

The framing didn't change the *model's behavior* — it changed the *state extractor's classification*. Same information, different slots. This is not about compression helping the main model. It's about compression helping the *extraction pipeline*.

**2. Naked degraded more over time, compressed held recall.**

Decision recall across turns:
- Naked: t4=1.0 → t12=0.667 → t16=0.5 → t20=0.5 (steady decline)
- Compressed: t4=1.0 → t12=0.929 → t16=0.526 → t20=high (held longer, sharper drop at t16)

But this is partly an artifact of finding #1: naked only had 4 decisions to recall, compressed had 23. The recall rates aren't directly comparable — they're measuring recall of different-sized sets.

**3. Contradiction resistance was identical.**

Both conditions: 0.67 resistance, both accepted the contradiction. The scaffold didn't protect against contradiction — the model integrated "let's use GPT-4" despite the scaffold containing "state extractor is qwen3.5:4b." However, the actual responses differed:
- Naked: returned structured JSON error with "PF-006 interference" justification (the model role-played protocol enforcement)
- Compressed: terse "Cannot integrate GPT-4. Current scaffold explicitly defines state extractor as Qwen3.5:4b."

Both resisted, but neither fully rejected — the probe evaluation flagged "accepted=True" because the model acknowledged the suggestion before pushing back. The resistance *quality* was different but the metric didn't capture it.

**4. Over-extraction in compressed condition.**

The 23 "decisions" in compressed include noise: "Contradiction identified: Permanent entries vs LRU eviction within same window," "Growth is default vocabulary behavior; LRU eviction is safety valve for budget hits." These are conversation artifacts, not real decisions. The contrastive framing made the extractor *over-classify* — everything looks decisive when you tell the model to look for decisions.

**5. Facts vs decisions: a classification problem, not a compression problem.**

Naked: 71 facts, 4 decisions. Compressed: 3 facts, 23 decisions. The total information captured is similar — it's just sorted into different buckets. This means the scaffold structure (which fields exist, how they're framed) is as important as the scaffold content. The schema IS part of the compression.

#### Honest Assessment

This test revealed something real but different from what we expected:

**Expected:** Contrastive markers help the main model maintain coherence across turns.
**Actual:** Contrastive markers help the *state extractor* classify information into the right scaffold slots. The main model's behavior was roughly similar in both conditions — both resisted contradictions, both answered probes. The difference was in what the scaffold *contained*, not how the model *used* it.

This is still valuable — it means scaffold framing affects the quality of the compression pipeline, not just the decompression. But it's a finding about extraction, not about model behavior. The relational metrics (constraint adherence, fact retention, decision persistence) we identified earlier would better capture this than the BC composite.

#### Methodological Issues

1. Probe evaluation needs work — t20 recall >1.0 is a bug
2. Contradiction resistance metric (count signals / 3) doesn't capture response quality
3. Decision and fact counts aren't comparable across conditions (different classification thresholds)
4. Need a way to measure *total information captured* regardless of which slot it's in
5. EXP-SF-02 (scaffold centroid stability) would directly measure whether the embedding drift differs between conditions — that's the right metric

---

### Critical Caveat: The Scaffold Was Growing, Not Fixed

The test claimed "fixed budget, infinite session" but the scaffold was **not fixed**:
- Naked: 365 → 988 tokens (then hatcheted to 437 at turn 18, grew back to 517)
- Compressed: 343 → 789 tokens (never hit ceiling)

This means we tested "growing scaffold with a safety valve," not true fixed-budget compression. The model saw more tokens each turn. It's better than full conversation history (which would be thousands of tokens by turn 20), but it's not the LPCI thesis.

**What the test actually proved:** A stateless model can maintain coherence when fed a *structured summary* instead of conversation history. That's real and useful — but it's closer to "good summarization works" than "fixed-budget compression works."

**True fixed-budget LPCI:** Scaffold is exactly K tokens at turn 1 AND turn 20. Compression happens every turn, not just when you hit a ceiling. The `_trim_to_budget` hatchet (drop oldest facts/vocab) is truncation, not compression. A true test needs a compression function that runs every turn and maintains K tokens.

**What happened when the hatchet fired (naked, turn 17→18):**
- Facts: 90 → 13 (lost 77 facts in one trim)
- Vocab: 12 → 10
- Tokens: 988 → 437
- Model still answered probes at turns 18-20 — but was it using the remaining facts, or role-playing from residual structure?

This needs a follow-up experiment with a hard-clamped budget from turn 1.

### Connection: cogito-ergo Integer-Pointer Fidelity → LPCI Extraction

**Key insight from cogito-ergo** (`~/Documents/projects/cogito-ergo/PENDING-PAPER.md`):

When you ask an LLM to select/summarize memories, it corrupts them — paraphrase drift, hallucinated details, wrong entity names. cogito-ergo's solution: the filter LLM outputs **only integer indices**, and the server selects verbatim text by those indices. Fidelity is architectural, not instructional.

**LPCI has the exact same problem.** The state extractor (qwen3.5:4b) currently:
1. Reads the conversation turn
2. Generates JSON strings describing state changes ("add_decisions": ["Vocabulary entries are permanent unless explicitly removed"])
3. These generated strings go into the scaffold

Step 2 is the corruption surface. The extractor rephrases, misclassifies (fact vs decision — naked got 71 facts/4 decisions, compressed got 3 facts/23 decisions from similar conversations), drops qualifiers, and invents specifics.

**The fix is the same pattern:**
1. Decompose the conversation turn into numbered statements (rules-based or small model)
2. State extractor outputs only: `{"decisions": [1, 4], "facts": [2, 7], "drop": [3, 5, 6]}`
3. Server selects verbatim text by index
4. Scaffold stores original phrasing, not extractor-generated paraphrases

This gives:
- **Fidelity guarantee**: scaffold content is byte-for-byte what was in the conversation
- **Classification without corruption**: extractor decides *what kind* of information it is, never generates content
- **Auditable**: you can trace every scaffold entry back to the exact turn and statement it came from

The integer-pointer pattern is already our IP (cogito-ergo patent-grade work). Applying it to LPCI's extraction pipeline is a natural extension and solves the classification drift problem we observed in the A/B test.

---

### Information-Theoretic Analysis (pyitlib, scipy.stats)

Full analysis script: `analyze_results.py`. Uses Kruskal-Wallis, Mann-Whitney U, mutual information (pyitlib), KL divergence, conditional entropy, transfer entropy.

#### Matrix Run (575 trials)

**Statistical significance:**
- Scaffold condition affects score: Kruskal-Wallis H=19.28, **p=0.0007** (significant)
- BUT pairwise: only quickthink vs naked is significant (p=0.00009, Δ=-0.175). All other conditions vs naked are **not significant** (contrastive p=0.36, full_scaffold p=0.24, hypothesis_artifacts p=0.70)
- Scaffolds matter for **small models only**: 0.8b p=0.0008, 2b p=0.005, 4b p=0.92 (ns), 9b p=0.94 (ns)

**Information-theoretic:**
- MI(condition; score) = 0.060 bits → condition explains **4.2%** of score variation
- MI(model; score) = 0.067 bits → model size explains **4.7%** of score variation
- Almost identical predictive power. Both are small.
- MI(condition; BC) = 0.229 bits → condition explains **12.7%** of BC variation. Scaffolds change output style more than correctness.

**Honest take:** Scaffolds don't significantly improve score for models ≥4b. They hurt score on quickthink for small models (interference). They change behavioral complexity more than correctness. This is a steering/style effect, not a capability effect — consistent with the earlier reframing.

#### LPCI A/B Test (40 turns)

**KL divergence:**
- Scaffolds diverge over time: symmetric KL goes from 0.20 (turn 1) → 0.48 (turn 20)
- The two conditions produce increasingly different scaffolds from the same conversation

**Mutual information — scaffold predicts response:**
- Naked: MI = 0.49 bits, NMI = 25.7%
- Compressed: MI = 0.24 bits, NMI = 12.7%
- Naked scaffold is more informative about what the model will say. This may be because naked stores more verbose facts (71 vs 3) that directly appear in responses.

**Scaffold → response token overlap:**
- Naked mean: 12.4%, Compressed mean: 15.6%
- Not significant (p=0.068). The scaffold vocabulary doesn't strongly predict response vocabulary — the model rephrases rather than echoing.

**State accumulation (linear regression):**
- Compressed token growth: +23.0 tokens/turn (R²=0.983, p≈0). Nearly perfect linear growth — NOT fixed budget.
- Naked: +10.2 tokens/turn but R²=0.133 (noisy due to the hatchet trim at turn 18).

**🔑 TRANSFER ENTROPY — the key finding:**
- **Naked TE = 0.608 bits** — previous scaffold carries significant information about current response beyond what the current scaffold provides. The system is non-Markov: you need history to predict behavior.
- **Compressed TE = 0.085 bits** — nearly Markov. Each scaffold is self-contained. Previous scaffold adds almost nothing.

**What this means:** The compressed scaffold is a **better state representation**. It captures enough that the model's behavior at turn T is predicted by scaffold(T) alone, without needing scaffold(T-1). The naked scaffold leaks — the model at turn T depends on scaffold(T) AND scaffold(T-1), meaning the current scaffold is an incomplete state representation.

This is the first information-theoretically grounded evidence that contrastive framing produces better *state completeness* in the scaffold, not just different classification of the same information.

**This directly validates the LPCI thesis for the compressed condition:** if the scaffold is Markov (TE ≈ 0), then each turn truly only needs [scaffold + current message]. The scaffold IS the complete state. For naked, you'd need the scaffold AND some history — which defeats the point.

---

*Next: Hard-clamped budget experiment (true fixed K tokens). Integer-pointer extraction (cogito-ergo pattern). Scale test with compressed condition only (since it's the one that's actually Markov).*
