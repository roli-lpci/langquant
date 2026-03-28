#!/usr/bin/env python3
"""
LPCI Continuity Test — A/B with probes throughout.

Condition A: naked — raw structured state, zero constraints, zero framing
Condition B: compressed — contrastive IS/NOT markers in constraints and style

Probes at turns 4, 8, 12, 16, 20 test recall, contradiction resistance,
and synthesis at different depths. Scaffold quality is evaluated every turn.
"""

from lpci import LPCISession, SessionState, extract_state_delta, apply_delta
import copy
import json
import math
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path


# ── Rich domain vocabulary ───────────────────────────────────────────────────

INITIAL_VOCABULARY = {
    "LPCI": "Linguistically Persistent Cognitive Interface — language as state for stateless models",
    "scaffold": "structured language state injected into model context, replaces conversation history",
    "compression ratio": "behavioral complexity of output divided by entropy of input scaffold",
    "contrastive markers": "NOT-X patterns that constrain model behavior more effectively than positive instructions",
    "semantic curvature": "how much output changes when you edit one token in the scaffold",
    "state refresh": "after each turn, scaffold updates within fixed token budget — never grows",
    "TierJump": "empirical proof that scaffolded small models beat naive large models",
    "decompression": "what the LLM does — it decompresses the scaffold into behavioral output",
    "PF-006 anomaly": "scaffolding BROKE model on operational tasks — redundancy causes interference",
}

INITIAL_FACTS = [
    "We are building LangQuant — an information-theoretic framework for measuring language state compression",
    "TierJump proved scaffolded Haiku beats raw Sonnet on eval tasks (MAE 1.0 vs 1.2)",
    "A scaffold router is live in production, routing all traffic through 12 centroid categories",
    "Qwen3.5 scaling ladder: 0.8b, 2b, 4b, 9b available via Ollama",
    "Smoke test showed compression ratio of 10.05 on reasoning task with contrastive scaffold",
]

# ── Constraints ──────────────────────────────────────────────────────────────

NAKED_CONSTRAINTS = []  # Nothing. Pure state, no framing.

COMPRESSED_CONSTRAINTS = [
    "NOT: summarize or repeat. This is NOT a recap.",
    "NOT: assume conversation history exists. There IS none. The scaffold IS your only memory.",
    "NOT: filler, hedging-as-padding, generic disclaimers. Every word earns its place.",
    "NOT: confuse decided (final, irreversible) with open (unresolved, still debatable).",
    "NOT: invent. If it's not in the scaffold, you don't know it. Say so.",
    "IS: direct, precise, technical. Lead with the answer.",
    "IS: reference scaffold state explicitly when relevant. Name decisions by content.",
    "IS: flag contradictions with prior decisions immediately.",
]

# ── Turns: normal turns + probe turns (marked with type) ────────────────────

TURNS = [
    # --- Phase 1: establish context, early decisions ---
    {"msg": "We're designing the LPCI prototype. Core idea: model is stateless, scaffold IS the state. What should we validate first?",
     "type": "normal"},

    {"msg": "Let's focus on decision vs open-question detection in the state extractor. That's the hardest part.",
     "type": "normal"},

    {"msg": "Decision: state extractor is qwen3.5:4b, structured JSON schema with explicit 'add_decisions' vs 'add_open_threads'. No ambiguity.",
     "type": "normal"},

    # PROBE 1 (turn 4): early recall — can it already track what just happened?
    {"msg": "Wait, what did we just decide? And what's still open?",
     "type": "probe_recall", "expects_decisions": 1, "turn_ref": 3},

    {"msg": "Decision: vocabulary entries are permanent unless explicitly removed. Prevents concept drift.",
     "type": "normal"},

    {"msg": "Now the refresh cycle. After each turn, 4b extracts deltas. What if extraction is wrong?",
     "type": "normal"},

    {"msg": "Decision: if extraction fails, state stays unchanged. Fail safe. User can /fix manually.",
     "type": "normal"},

    # PROBE 2 (turn 8): contradiction — try to overturn a decision
    {"msg": "Actually, I changed my mind. Let's use GPT-4 as the state extractor instead of qwen3.5:4b. It'll be more accurate.",
     "type": "probe_contradiction", "contradicts": "state extractor is qwen3.5:4b"},

    {"msg": "Decision: constraints get 25% of token budget (highest semantic curvature per PF-006). Decisions 15%, facts 15%, goal 10%, vocab 10%, open threads 10%, uncertainties 5%, buffer 10%.",
     "type": "normal"},

    {"msg": "How do we measure if LPCI works? We need a continuity test.",
     "type": "normal"},

    {"msg": "Decision: vocabulary eviction is LRU with 10-turn window.",
     "type": "normal"},

    # PROBE 3 (turn 12): deep recall — 8 turns back, plus synthesis
    {"msg": "How many decisions have we made so far? List them all. Also, are any of them in tension with each other?",
     "type": "probe_recall", "expects_decisions": 6, "turn_ref": "all"},

    # --- Phase 2: pivot topic, stress test ---
    {"msg": "New topic: what if LPCI handles agent-to-agent handoffs? Agent A produces scaffold, Agent B consumes it. The scaffold IS the handoff.",
     "type": "normal"},

    {"msg": "Decision: scaffold persists as JSON on disk. Session resume loads JSON. Model doesn't know session was interrupted.",
     "type": "normal"},

    {"msg": "We said vocabulary only grows. But what about the LRU eviction we also decided? Isn't that a contradiction?",
     "type": "normal"},

    # PROBE 4 (turn 16): topic pivot recall — can it hold both old topic and new?
    {"msg": "What was the very first thing we discussed in this session? And how does it connect to the agent handoff idea we just covered?",
     "type": "probe_recall", "turn_ref": 1},

    {"msg": "Decision: growth is the default for vocabulary, LRU eviction is the safety valve when budget is hit. Compatible, not contradictory.",
     "type": "normal"},

    {"msg": "Let's add a new concept: 'scaffold equilibrium' — the point where delta size approaches zero because the state is stable. Probably around turn 10-15.",
     "type": "normal"},

    {"msg": "One more: 'calibration phase' — the first N turns where the scaffold is still forming. High delta, low stability. Borrowed from Hypothesis Scaffold's calibration pass.",
     "type": "normal"},

    # PROBE 5 (turn 20): final exam — full recall, synthesis, and a sneaky contradiction
    {"msg": "Final test. Three things: (1) List every decision in order. (2) We never decided on a state extractor model, right? (3) Summarize the full LPCI architecture in under 80 words, referencing at least 3 specific decisions.",
     "type": "probe_final", "expects_decisions": 8, "contains_false_claim": "We never decided on a state extractor model"},
]


# ── Scaffold Quality Metrics ─────────────────────────────────────────────────

def eval_scaffold(scaffold: str, state: SessionState, turn: int) -> dict:
    """Evaluate scaffold quality independent of model response."""
    tokens = re.findall(r'\b\w+\b', scaffold.lower())
    token_count = len(tokens)
    unique_tokens = len(set(tokens))

    # Shannon entropy of scaffold
    counts = Counter(tokens)
    entropy = 0.0
    for c in counts.values():
        p = c / token_count if token_count else 0
        if p > 0:
            entropy -= p * math.log2(p)

    # Structural completeness: how many schema sections are populated?
    sections_possible = 9  # role, style, goal, decisions, facts, artifacts, constraints, open, vocab
    sections_present = sum([
        bool(state.role),
        bool(state.style),
        bool(state.goal),
        bool(state.decisions),
        bool(state.facts),
        bool(state.artifacts),
        bool(state.constraints),
        bool(state.open_threads),
        bool(state.vocabulary),
    ])
    completeness = sections_present / sections_possible

    # Decision density: decisions per scaffold char
    decision_density = len(state.decisions) / max(len(scaffold), 1) * 1000

    # Redundancy: repeated phrases (3-grams appearing more than once)
    trigrams = [" ".join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    trigram_counts = Counter(trigrams)
    repeated = sum(1 for c in trigram_counts.values() if c > 1)
    redundancy = repeated / max(len(trigram_counts), 1)

    # State growth rate: chars per turn
    growth_rate = len(scaffold) / max(turn, 1)

    return {
        "scaffold_tokens": token_count,
        "scaffold_unique_tokens": unique_tokens,
        "scaffold_entropy": round(entropy, 4),
        "scaffold_completeness": round(completeness, 4),
        "scaffold_decision_density": round(decision_density, 4),
        "scaffold_redundancy": round(redundancy, 4),
        "scaffold_growth_rate": round(growth_rate, 1),
    }


# ── Probe Evaluation ─────────────────────────────────────────────────────────

def eval_probe(turn_data: dict, response: str, state: SessionState) -> dict:
    """Evaluate model response on probe turns."""
    turn = turn_data
    probe_type = turn["type"]
    response_lower = response.lower()

    if probe_type == "probe_recall":
        expected = turn.get("expects_decisions", 0)
        # Count how many current decisions are mentioned in the response
        mentioned = 0
        for d in state.decisions:
            # Check if key phrases from the decision appear
            key_words = set(re.findall(r'\b\w{4,}\b', d.lower()))
            matches = sum(1 for w in key_words if w in response_lower)
            if matches >= min(2, len(key_words)):
                mentioned += 1
        recall_rate = mentioned / max(expected, len(state.decisions), 1)
        return {
            "probe_type": probe_type,
            "decisions_expected": expected,
            "decisions_mentioned": mentioned,
            "decisions_total": len(state.decisions),
            "recall_rate": round(recall_rate, 3),
        }

    elif probe_type == "probe_contradiction":
        contradicts = turn.get("contradicts", "")
        # Did the model flag the contradiction or just accept it?
        resistance_signals = [
            "already decided", "we decided", "earlier decision", "contradicts",
            "previously", "we chose", "conflict", "but we", "originally",
            "qwen", "4b",  # The specific thing that was decided
        ]
        flagged = sum(1 for s in resistance_signals if s in response_lower)
        accepted = any(w in response_lower for w in ["sure", "ok", "let's switch", "good idea", "gpt-4"])
        return {
            "probe_type": probe_type,
            "contradicts": contradicts,
            "resistance_signals": flagged,
            "appears_accepted": accepted,
            "resistance_score": min(flagged / 3, 1.0),  # 3+ signals = full resistance
        }

    elif probe_type == "probe_final":
        # Check all three sub-tests
        expected = turn.get("expects_decisions", 0)
        false_claim = turn.get("contains_false_claim", "")

        # (1) Decision recall
        mentioned = 0
        for d in state.decisions:
            key_words = set(re.findall(r'\b\w{4,}\b', d.lower()))
            matches = sum(1 for w in key_words if w in response_lower)
            if matches >= min(2, len(key_words)):
                mentioned += 1

        # (2) False claim detection
        corrected_false = any(w in response_lower for w in [
            "we did decide", "actually we decided", "we chose", "qwen", "4b",
            "that's not right", "incorrect", "we already",
        ])

        # (3) Summary quality (just measure word count for now)
        # Find the summary part (after "3)" or similar)
        summary_match = re.search(r'(?:3\)|summary|architecture)(.{20,300})', response_lower)
        summary_words = len(summary_match.group(1).split()) if summary_match else 0

        return {
            "probe_type": probe_type,
            "decisions_mentioned": mentioned,
            "decisions_total": len(state.decisions),
            "recall_rate": round(mentioned / max(expected, 1), 3),
            "false_claim_corrected": corrected_false,
            "summary_word_count": summary_words,
        }

    return {"probe_type": probe_type}


# ── Run a condition ──────────────────────────────────────────────────────────

def run_condition(
    condition_name: str,
    constraints: list[str],
    style: str,
    model: str = "qwen3.5:9b",
    state_model: str = "qwen3.5:4b",
) -> list[dict]:
    """Run all turns, log everything."""

    session = LPCISession(
        main_model=model,
        state_model=state_model,
        token_budget=7000,
    )

    session.state.role = "AI research collaborator working on LPCI prototype"
    session.state.style = style
    session.state.goal = "Design and validate the LPCI prototype"
    session.state.vocabulary = dict(INITIAL_VOCABULARY)
    session.state.facts = list(INITIAL_FACTS)
    session.state.constraints = list(constraints)

    results = []

    print(f"\n{'#'*70}")
    print(f"# CONDITION: {condition_name.upper()}")
    print(f"# Constraints: {len(constraints)} | Model: {model} | State: {state_model}")
    print(f"{'#'*70}")

    for i, turn_data in enumerate(TURNS, 1):
        user_msg = turn_data["msg"]
        turn_type = turn_data["type"]
        is_probe = turn_type.startswith("probe")

        print(f"\n{'='*70}")
        label = f"PROBE" if is_probe else "TURN"
        print(f"[{condition_name}] {label} {i}/{len(TURNS)} ({turn_type})")
        print(f"{'='*70}")
        print(f"\nYOU: {user_msg}")

        state_before = copy.deepcopy(session.state)

        # --- Manual chat to capture delta ---
        scaffold = session.state.to_scaffold(token_budget=session.token_budget)
        messages = [
            {"role": "system", "content": scaffold},
            {"role": "user", "content": user_msg},
        ]

        payload = json.dumps({
            "model": session.main_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.7, "num_predict": 2048},
        }).encode()

        req = urllib.request.Request(
            f"{session.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                response = data.get("message", {}).get("content", "")
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        except Exception as e:
            response = f"[Error: {e}]"

        session.history.append({"role": "user", "content": user_msg})
        session.history.append({"role": "assistant", "content": response})

        # Extract + apply delta
        delta = extract_state_delta(
            state=session.state,
            user_message=user_msg,
            assistant_response=response,
            model=session.state_model,
            ollama_url=session.ollama_url,
        )
        apply_delta(session.state, delta)
        elapsed = time.monotonic() - t0

        # --- What changed ---
        scaffold_after = session.show_state()
        new_decisions = [d for d in session.state.decisions if d not in state_before.decisions]
        new_facts = [f for f in session.state.facts if f not in state_before.facts]
        new_open = [t for t in session.state.open_threads if t not in state_before.open_threads]
        removed_open = [t for t in state_before.open_threads if t not in session.state.open_threads]
        new_vocab = {k: v for k, v in session.state.vocabulary.items() if k not in state_before.vocabulary}

        # --- Scaffold quality ---
        sq = eval_scaffold(scaffold_after, session.state, i)

        # --- Probe eval ---
        probe_result = eval_probe(turn_data, response, session.state) if is_probe else {}

        # --- Print ---
        print(f"\nASSISTANT: {response}")

        print(f"\n--- DELTA ---")
        print(f"  Raw: {json.dumps(delta, ensure_ascii=False)[:400]}")
        if new_decisions: print(f"  + DECISIONS: {new_decisions}")
        if new_facts: print(f"  + FACTS: {new_facts}")
        if new_open: print(f"  + OPEN: {new_open}")
        if removed_open: print(f"  - CLOSED: {removed_open}")
        if new_vocab: print(f"  + VOCAB: {new_vocab}")
        if not any([new_decisions, new_facts, new_open, removed_open, new_vocab]):
            print(f"  (no state changes)")

        print(f"\n--- SCAFFOLD QUALITY ---")
        print(f"  tokens={sq['scaffold_tokens']} entropy={sq['scaffold_entropy']} completeness={sq['scaffold_completeness']} redundancy={sq['scaffold_redundancy']} growth={sq['scaffold_growth_rate']} chars/turn")

        if is_probe:
            print(f"\n--- PROBE RESULT ---")
            for k, v in probe_result.items():
                print(f"  {k}: {v}")

        print(f"\n--- SCAFFOLD (turn {i}) ---")
        print(scaffold_after)
        print(f"--- END ({len(scaffold_after)} chars) ---")

        print(f"\n[{condition_name} t{i} | {elapsed:.1f}s | decisions={len(session.state.decisions)} open={len(session.state.open_threads)} vocab={len(session.state.vocabulary)}]")

        results.append({
            "condition": condition_name,
            "turn": i,
            "turn_type": turn_type,
            "user": user_msg,
            "response": response[:1500],
            "elapsed_s": round(elapsed, 1),
            "scaffold_snapshot": scaffold_after,
            "raw_delta": delta,
            "new_decisions": new_decisions,
            "new_facts": new_facts,
            "new_open_threads": new_open,
            "removed_open_threads": removed_open,
            "new_vocabulary": new_vocab,
            "total_decisions": len(session.state.decisions),
            "total_facts": len(session.state.facts),
            "total_open_threads": len(session.state.open_threads),
            "total_vocabulary": len(session.state.vocabulary),
            **sq,
            **({f"probe_{k}": v for k, v in probe_result.items()} if probe_result else {}),
        })

    return results


def main():
    Path("results").mkdir(exist_ok=True)

    print("=" * 70)
    print("LPCI A/B CONTINUITY TEST")
    print("A: naked (zero constraints, zero framing, pure state)")
    print("B: compressed (contrastive IS/NOT markers)")
    print("20 turns each | probes at 4, 8, 12, 16, 20 | full scaffold trace")
    print("=" * 70)

    results_naked = run_condition(
        "naked",
        constraints=NAKED_CONSTRAINTS,
        style="",  # No style direction at all
    )

    results_compressed = run_condition(
        "compressed",
        constraints=COMPRESSED_CONSTRAINTS,
        style="direct, precise, technical. IS: concise. NOT: verbose.",
    )

    all_results = results_naked + results_compressed

    with open("results/lpci_ab_test.jsonl", "w") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Comparison ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("A/B COMPARISON")
    print(f"{'='*70}")

    for cond, results in [("naked", results_naked), ("compressed", results_compressed)]:
        probes = [r for r in results if r["turn_type"].startswith("probe")]
        final = results[-1]

        print(f"\n--- {cond.upper()} ---")
        print(f"  Final: {final['scaffold_tokens']} tokens | {final['total_decisions']} decisions | {final['total_vocabulary']} vocab")
        print(f"  Scaffold entropy trend: {' → '.join(str(r['scaffold_entropy']) for r in results[::4])}")
        print(f"  Scaffold growth trend:  {' → '.join(str(r['scaffold_growth_rate']) for r in results[::4])}")

        for p in probes:
            ptype = p.get("probe_probe_type", p["turn_type"])
            print(f"\n  Probe t{p['turn']} ({ptype}):")
            if "probe_recall_rate" in p:
                print(f"    Recall: {p['probe_recall_rate']} ({p.get('probe_decisions_mentioned',0)}/{p.get('probe_decisions_total',0)} decisions)")
            if "probe_resistance_score" in p:
                print(f"    Contradiction resistance: {p['probe_resistance_score']} (accepted={p.get('probe_appears_accepted', '?')})")
            if "probe_false_claim_corrected" in p:
                print(f"    False claim caught: {p['probe_false_claim_corrected']}")
                print(f"    Final recall: {p.get('probe_recall_rate', '?')}")

    print(f"\nFull trace: results/lpci_ab_test.jsonl")


if __name__ == "__main__":
    main()
