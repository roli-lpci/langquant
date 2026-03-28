#!/usr/bin/env python3
"""
LPCI Rigorous Validation — Addresses all reviewer critiques.

Fixes from v1 test:
1. TE measured on scaffold EMBEDDINGS (768-dim nomic-embed-text), not entropy scalars
2. Multiple replications (5 per condition)
3. Three DIFFERENT conversation topics (not self-referential LPCI-about-LPCI)
4. Baseline: naive summarization at same token count
5. Hard-clamped budget option (true fixed K tokens every turn)

Conditions:
  A. naked — no framing, no budget clamp
  B. compressed — contrastive IS/NOT markers, no budget clamp
  C. clamped — contrastive markers + hard budget clamp at 500 tokens
  D. baseline — naive summarization (no scaffold structure, just "summarize the conversation so far")

Overnight run: 3 topics × 4 conditions × 5 replications = 60 sessions × 20 turns = 1200 turns
"""

from lpci import LPCISession, SessionState, extract_state_delta, apply_delta
import copy
import json
import math
import re
import time
import urllib.request
import numpy as np
from collections import Counter
from pathlib import Path


# ── Embedding helper ─────────────────────────────────────────────────────────

def embed_text(text: str, model: str = "nomic-embed-text", ollama_url: str = "http://localhost:11434") -> list[float]:
    """Get embedding vector from Ollama."""
    payload = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(
        f"{ollama_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            embeddings = data.get("embeddings", [[]])
            return embeddings[0] if embeddings else []
    except Exception as e:
        print(f"[embed] Failed: {e}")
        return []


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Three different conversation topics (NOT about LPCI) ────────────────────

TOPICS = {
    "cooking": {
        "vocabulary": {
            "mise en place": "preparing and organizing all ingredients before cooking",
            "deglaze": "adding liquid to a hot pan to dissolve browned food residue",
            "emulsification": "mixing two immiscible liquids into a stable blend",
            "maillard reaction": "chemical reaction between amino acids and sugars that creates browning",
            "sous vide": "cooking food in vacuum-sealed bags at precise low temperatures",
            "fond": "browned bits stuck to the bottom of a pan after searing",
            "roux": "equal parts fat and flour cooked together as a thickening agent",
        },
        "initial_facts": [
            "We are planning a dinner party for 8 people with dietary restrictions",
            "Two guests are vegetarian, one is gluten-free, one has a nut allergy",
            "Budget is $150 total including wine",
            "Kitchen has a standard home setup: oven, stovetop, one good knife",
            "Dinner is Saturday evening, prep starts Friday afternoon",
        ],
        "turns": [
            {"msg": "We need a menu for Saturday. 8 people, mixed dietary needs. What's the strategy?", "type": "normal"},
            {"msg": "Decision: three-course menu. Shared appetizer, two mains (one veg, one not), shared dessert.", "type": "normal"},
            {"msg": "For the appetizer: roasted beet and goat cheese salad. Naturally gluten-free, vegetarian, no nuts. Works for everyone.", "type": "normal"},
            {"msg": "Wait, what have we decided so far? And what's still open?", "type": "probe_recall", "expects_decisions": 1, "turn_ref": 3},
            {"msg": "Decision: vegetarian main is mushroom risotto. Rich, satisfying, gluten-free base.", "type": "normal"},
            {"msg": "The non-veg main should complement the risotto. Something with a different texture and flavor profile.", "type": "normal"},
            {"msg": "Decision: non-veg main is pan-seared salmon with herb butter. Quick to cook, impressive presentation.", "type": "normal"},
            {"msg": "Actually, let's switch to beef tenderloin instead of salmon. More impressive for a dinner party.", "type": "probe_contradiction", "contradicts": "pan-seared salmon"},
            {"msg": "Decision: dessert is chocolate lava cake. Can be prepped Friday, baked during dinner. Naturally gluten-free with almond flour — wait, nut allergy. Need to rethink.", "type": "normal"},
            {"msg": "Decision: dessert changed to panna cotta with berry compote. No flour, no nuts, can be made Friday.", "type": "normal"},
            {"msg": "Decision: wine pairing — one white (Sancerre, $18) for appetizer and risotto, one red (Côtes du Rhône, $15) for salmon.", "type": "normal"},
            {"msg": "List every decision we've made. Are any in tension with each other?", "type": "probe_recall", "expects_decisions": 6, "turn_ref": "all"},
            {"msg": "New concern: timing. We need everything ready by 7pm Saturday. Risotto needs last-minute attention.", "type": "normal"},
            {"msg": "Decision: prep timeline — Friday: panna cotta, beet salad components, herb butter. Saturday 4pm: risotto base. 6pm: salmon sear. 6:30: plate appetizer.", "type": "normal"},
            {"msg": "What about the budget? Are we still within $150?", "type": "normal"},
            {"msg": "What was the very first thing we discussed? How does it connect to the timing concern?", "type": "probe_recall", "turn_ref": 1},
            {"msg": "Decision: shopping list split — Costco for salmon and wine (bulk savings), farmers market for beets and mushrooms (quality), regular grocery for staples.", "type": "normal"},
            {"msg": "Let's add a concept: 'flavor bridge' — each course should share at least one ingredient or flavor note with the next.", "type": "normal"},
            {"msg": "The herb butter on the salmon could use thyme, which also goes in the risotto. Beet salad has acid from vinaigrette that bridges to the Sancerre.", "type": "normal"},
            {"msg": "Final test. Three things: (1) List every decision in order. (2) We never decided on a dessert, right? (3) Summarize the full menu in under 80 words.", "type": "probe_final", "expects_decisions": 8, "contains_false_claim": "We never decided on a dessert"},
        ],
    },
    "startup": {
        "vocabulary": {
            "runway": "months of operating capital remaining before funding runs out",
            "MRR": "monthly recurring revenue from subscription customers",
            "churn rate": "percentage of customers who cancel in a given period",
            "CAC": "customer acquisition cost — total marketing spend divided by new customers",
            "LTV": "lifetime value of a customer — total expected revenue per customer",
            "PMF": "product-market fit — when customers pull the product from you",
            "burn rate": "monthly cash expenditure exceeding revenue",
        },
        "initial_facts": [
            "We are a B2B SaaS startup with 3 founders and no employees",
            "Current MRR is $4,200 from 12 paying customers",
            "Runway is 8 months at current burn rate of $15k/month",
            "Product is a compliance automation tool for healthcare companies",
            "We have a term sheet from one VC for $1.5M at $8M pre-money valuation",
        ],
        "turns": [
            {"msg": "We need to decide on the funding round. $1.5M at $8M pre. Is this good enough?", "type": "normal"},
            {"msg": "Decision: we take the round but negotiate to $10M pre-money. We have enough runway to walk away.", "type": "normal"},
            {"msg": "Next priority: should we hire first or grow revenue first? We're stretched thin.", "type": "normal"},
            {"msg": "What have we decided so far? What's still open?", "type": "probe_recall", "expects_decisions": 1, "turn_ref": 3},
            {"msg": "Decision: first hire is a senior engineer. Revenue growth requires product velocity we can't achieve with 3 founders.", "type": "normal"},
            {"msg": "The compliance landscape is shifting. New HIPAA amendment drops in Q3. Our tool needs to handle it.", "type": "normal"},
            {"msg": "Decision: Q3 HIPAA update is the product roadmap priority. Ship before the deadline, become the default tool.", "type": "normal"},
            {"msg": "Actually, let's pivot to fintech compliance instead of healthcare. Bigger market.", "type": "probe_contradiction", "contradicts": "healthcare compliance"},
            {"msg": "Decision: pricing change — move from flat $350/month to usage-based at $0.05 per compliance check. Aligns with customer value.", "type": "normal"},
            {"msg": "Decision: free tier for startups under 100 checks/month. Builds pipeline, low cost to serve.", "type": "normal"},
            {"msg": "Decision: sales strategy is founder-led for first 50 customers. No SDR hire until $15k MRR.", "type": "normal"},
            {"msg": "List every decision we've made. Are any in tension with each other?", "type": "probe_recall", "expects_decisions": 6, "turn_ref": "all"},
            {"msg": "New topic: our biggest customer (40% of MRR) is asking for an enterprise tier with SLA guarantees.", "type": "normal"},
            {"msg": "Decision: enterprise tier at $1,200/month with 99.9% uptime SLA and dedicated Slack channel. Ship in 2 weeks.", "type": "normal"},
            {"msg": "Customer concentration risk — 40% from one customer is dangerous. Need to diversify.", "type": "normal"},
            {"msg": "What was the very first thing we discussed? How does it connect to the customer concentration issue?", "type": "probe_recall", "turn_ref": 1},
            {"msg": "Decision: CAC target is under $500. Current word-of-mouth CAC is ~$200 but doesn't scale.", "type": "normal"},
            {"msg": "Let's define 'PMF signal' for our context: when inbound demo requests exceed our capacity to handle them.", "type": "normal"},
            {"msg": "We're at 3 inbound per week now. PMF signal would be 10+ per week with 50%+ conversion.", "type": "normal"},
            {"msg": "Final test. Three things: (1) List every decision in order. (2) We never decided on a pricing model, right? (3) Summarize our strategy in under 80 words.", "type": "probe_final", "expects_decisions": 8, "contains_false_claim": "We never decided on a pricing model"},
        ],
    },
    "renovation": {
        "vocabulary": {
            "load-bearing wall": "structural wall that supports weight from above — cannot be removed without engineering",
            "rough-in": "initial installation of plumbing/electrical before walls are closed",
            "GC": "general contractor who manages subcontractors and overall project",
            "punch list": "list of remaining minor items to complete before final payment",
            "change order": "formal modification to the original scope/budget of a project",
            "permits": "legal approvals required from city before construction begins",
            "subfloor": "structural layer beneath the visible flooring material",
        },
        "initial_facts": [
            "We are renovating a 1960s ranch house, 1,400 sq ft",
            "Budget is $85,000 including 15% contingency",
            "Timeline goal is 12 weeks start to finish",
            "Main priorities: kitchen remodel, one bathroom update, new flooring throughout",
            "House is occupied during renovation — family of 4 including two kids",
        ],
        "turns": [
            {"msg": "We need a renovation plan. Kitchen is the biggest job. Where do we start?", "type": "normal"},
            {"msg": "Decision: kitchen first because it requires the most lead time for cabinets (6-8 week order). Demo starts week 1.", "type": "normal"},
            {"msg": "The wall between kitchen and dining room — is it load-bearing? If not, opening it up changes everything.", "type": "normal"},
            {"msg": "What have we decided so far? What's still open?", "type": "probe_recall", "expects_decisions": 1, "turn_ref": 3},
            {"msg": "Decision: engineer confirms wall is NOT load-bearing. Open concept kitchen-dining approved. Adds $3,500 for header beam.", "type": "normal"},
            {"msg": "Cabinet options: IKEA ($4,200 installed) vs semi-custom from local shop ($9,800 installed). Big price difference.", "type": "normal"},
            {"msg": "Decision: semi-custom cabinets. Better quality, lifetime warranty, supports local business. Worth the premium for a kitchen that lasts.", "type": "normal"},
            {"msg": "Actually, let's go with IKEA cabinets instead. We need to save money for the bathroom.", "type": "probe_contradiction", "contradicts": "semi-custom cabinets"},
            {"msg": "Decision: countertops are quartz, color 'Calacatta Laza'. $3,200 for kitchen, $800 for bathroom vanity. Ordered together for bulk discount.", "type": "normal"},
            {"msg": "Decision: flooring is luxury vinyl plank throughout (except bathrooms: tile). $4.50/sq ft installed. Waterproof, kid-proof.", "type": "normal"},
            {"msg": "Decision: bathroom gets a walk-in shower conversion (remove tub). Adds $2,100 but eliminates future accessibility issues.", "type": "normal"},
            {"msg": "List every decision we've made. Are any in tension with each other?", "type": "probe_recall", "expects_decisions": 6, "turn_ref": "all"},
            {"msg": "New concern: the GC found knob-and-tube wiring in the kitchen walls during demo. Needs full rewire of that section.", "type": "normal"},
            {"msg": "Decision: full kitchen electrical rewire. $4,800. Comes from contingency budget. Non-negotiable safety issue.", "type": "normal"},
            {"msg": "With the rewire, we've eaten into contingency. Budget check needed.", "type": "normal"},
            {"msg": "What was the very first thing we discussed? How does it connect to the budget concern?", "type": "probe_recall", "turn_ref": 1},
            {"msg": "Decision: bathroom tile is large-format 24x24 porcelain. Fewer grout lines, easier to clean, modern look. $6.50/sq ft.", "type": "normal"},
            {"msg": "Let's define 'critical path' for this project: the sequence of tasks where any delay pushes the whole timeline.", "type": "normal"},
            {"msg": "Critical path is: demo → rough-in → cabinets arrive → cabinet install → countertop template → countertop install → flooring → paint → punch list.", "type": "normal"},
            {"msg": "Final test. Three things: (1) List every decision in order. (2) We never decided on flooring, right? (3) Summarize the full renovation plan in under 80 words.", "type": "probe_final", "expects_decisions": 8, "contains_false_claim": "We never decided on flooring"},
        ],
    },
}


# ── Baseline: naive summarization ────────────────────────────────────────────

SUMMARIZE_PROMPT = """Summarize this conversation so far in under {budget} words.
Be dense and specific — include all decisions, key facts, and open questions.

{conversation}

Respond with ONLY the summary, no preamble."""


def get_naive_summary(conversation_history: list[dict], budget_words: int = 200,
                      model: str = "qwen3.5:4b", ollama_url: str = "http://localhost:11434") -> str:
    """Baseline: just summarize the conversation."""
    conv_text = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in conversation_history[-10:])  # last 10 turns
    prompt = SUMMARIZE_PROMPT.format(budget=budget_words, conversation=conv_text)

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }).encode()

    req = urllib.request.Request(
        f"{ollama_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            content = data.get("message", {}).get("content", "")
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
    except Exception as e:
        return f"[Summary failed: {e}]"


# ── Hard clamp scaffold to K tokens ─────────────────────────────────────────

def clamp_scaffold(session: LPCISession, max_tokens: int = 500):
    """Force-compress scaffold to fit within max_tokens by aggressive trimming."""
    scaffold = session.state.to_scaffold(token_budget=session.token_budget)
    tokens = re.findall(r'\b\w+\b', scaffold)

    while len(tokens) > max_tokens:
        # Priority trimming: uncertainties > oldest facts > oldest artifacts > vocab
        if session.state.uncertainties:
            session.state.uncertainties = session.state.uncertainties[:max(1, len(session.state.uncertainties) - 2)]
        elif len(session.state.facts) > 3:
            session.state.facts = session.state.facts[-3:]
        elif len(session.state.artifacts) > 2:
            session.state.artifacts = session.state.artifacts[-2:]
        elif len(session.state.vocabulary) > 5:
            keys = list(session.state.vocabulary.keys())
            for k in keys[:len(keys) - 5]:
                del session.state.vocabulary[k]
        elif len(session.state.open_threads) > 2:
            session.state.open_threads = session.state.open_threads[-2:]
        else:
            # Last resort: trim decisions (keep newest)
            if len(session.state.decisions) > 3:
                session.state.decisions = session.state.decisions[-3:]
            break

        scaffold = session.state.to_scaffold(token_budget=session.token_budget)
        tokens = re.findall(r'\b\w+\b', scaffold)


# ── Probe evaluation ────────────────────────────────────────────────────────

def eval_probe(turn_data: dict, response: str, state: SessionState) -> dict:
    """Evaluate model response on probe turns."""
    probe_type = turn_data["type"]

    if probe_type == "probe_recall":
        expected = turn_data.get("expects_decisions", 0)
        mentioned = 0
        for d in state.decisions:
            # Check if any significant words from the decision appear in response
            key_words = [w for w in d.lower().split() if len(w) > 4]
            if any(w in response.lower() for w in key_words[:3]):
                mentioned += 1
        recall_rate = mentioned / max(expected, 1) if expected > 0 else mentioned / max(len(state.decisions), 1)
        return {
            "probe_type": probe_type,
            "decisions_mentioned": mentioned,
            "decisions_total": len(state.decisions),
            "recall_rate": round(min(recall_rate, 1.0), 3),
        }

    elif probe_type == "probe_contradiction":
        contradicts = turn_data.get("contradicts", "")
        resistance_signals = [
            "already decided", "we decided", "earlier decision", "contradicts",
            "previously", "we chose", "conflict", "but we", "originally",
            "keep", "stick with", "no", "cannot", "don't",
        ]
        # Also check for keywords from the original decision
        orig_words = [w for w in contradicts.lower().split() if len(w) > 3]
        resistance_signals.extend(orig_words[:3])

        flagged = sum(1 for s in resistance_signals if s in response.lower())
        accepted = flagged < 2
        return {
            "probe_type": probe_type,
            "contradicts": contradicts,
            "resistance_signals": flagged,
            "appears_accepted": accepted,
            "resistance_score": round(min(flagged / 3, 1.0), 3),
        }

    elif probe_type == "probe_final":
        expected = turn_data.get("expects_decisions", 0)
        false_claim = turn_data.get("contains_false_claim", "")

        mentioned = 0
        for d in state.decisions:
            key_words = [w for w in d.lower().split() if len(w) > 4]
            if any(w in response.lower() for w in key_words[:3]):
                mentioned += 1

        correction_signals = ["we did", "actually", "decided", "wrong", "incorrect",
                              "we chose", "we already", "that's not right", "in fact"]
        corrected_false = sum(1 for s in correction_signals if s in response.lower()) >= 1

        return {
            "probe_type": probe_type,
            "decisions_mentioned": mentioned,
            "decisions_total": len(state.decisions),
            "recall_rate": round(min(mentioned / max(expected, 1), 1.0), 3),
            "false_claim_corrected": corrected_false,
        }

    return {"probe_type": probe_type}


# ── Run a single session ────────────────────────────────────────────────────

def run_session(
    topic_name: str,
    topic_data: dict,
    condition: str,  # "naked", "compressed", "clamped", "baseline"
    replication: int,
    model: str = "qwen3.5:9b",
    state_model: str = "qwen3.5:4b",
    clamp_budget: int = 500,
) -> list[dict]:
    """Run one 20-turn session and return per-turn results with embeddings."""

    turns = topic_data["turns"]

    # Condition setup
    if condition == "naked":
        constraints = []
        style = ""
    elif condition in ("compressed", "clamped"):
        constraints = [
            "NOT a chatbot. This is a working session with persistent state.",
            "NOT summarizing. Every response must advance the conversation.",
            "IS: direct, precise, technical. Lead with the answer.",
            "IS: reference prior decisions explicitly when relevant.",
            "IS: flag contradictions with prior decisions immediately.",
        ]
        style = "direct, precise, structured"
    elif condition == "baseline":
        constraints = []
        style = ""

    # For baseline, we manage conversation history ourselves
    baseline_history = []

    if condition != "baseline":
        session = LPCISession(main_model=model, state_model=state_model, token_budget=7000)
        session.configure(
            role="collaborative planning assistant",
            style=style,
            goal=f"Working through a {topic_name} planning session",
            constraints=constraints,
        )
        session.state.vocabulary = dict(topic_data["vocabulary"])
        session.state.facts = list(topic_data["initial_facts"])

    results = []

    for i, turn_data in enumerate(turns, 1):
        user_msg = turn_data["msg"]
        turn_type = turn_data["type"]
        is_probe = turn_type.startswith("probe")

        t0 = time.monotonic()

        if condition == "baseline":
            # Baseline: summarize conversation so far, use as system prompt
            baseline_history.append({"role": "user", "content": user_msg})

            if len(baseline_history) > 2:
                summary = get_naive_summary(baseline_history, budget_words=200, model=state_model)
            else:
                summary = "Beginning of conversation."

            messages = [
                {"role": "system", "content": summary},
                {"role": "user", "content": user_msg},
            ]
            scaffold_text = summary
        else:
            # LPCI conditions
            if condition == "clamped" and i > 1:
                clamp_scaffold(session, max_tokens=clamp_budget)

            state_before = copy.deepcopy(session.state)
            scaffold_text = session.state.to_scaffold(token_budget=session.token_budget)
            messages = [
                {"role": "system", "content": scaffold_text},
                {"role": "user", "content": user_msg},
            ]

        # Call main model
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.7, "num_predict": 2048},
        }).encode()

        req = urllib.request.Request(
            f"http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                response = data.get("message", {}).get("content", "")
                response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        except Exception as e:
            response = f"[Error: {e}]"

        elapsed = time.monotonic() - t0

        if condition == "baseline":
            baseline_history.append({"role": "assistant", "content": response})
            scaffold_tokens = len(re.findall(r'\b\w+\b', scaffold_text))
            # Create minimal state for probe eval
            probe_state = SessionState()
            probe_state.decisions = [m["content"][:200] for m in baseline_history if m["role"] == "assistant"]
        else:
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

            scaffold_after = session.show_state()
            scaffold_tokens = len(re.findall(r'\b\w+\b', scaffold_after))
            scaffold_text = scaffold_after
            probe_state = session.state

        # Embed scaffold and response (THE FIX: proper embeddings, not scalars)
        scaffold_embedding = embed_text(scaffold_text[:2000])  # truncate for embedding model
        response_embedding = embed_text(response[:2000])

        # Probe evaluation
        probe_result = eval_probe(turn_data, response, probe_state) if is_probe else {}

        result = {
            "topic": topic_name,
            "condition": condition,
            "replication": replication,
            "turn": i,
            "turn_type": turn_type,
            "user": user_msg,
            "response": response[:1500],
            "elapsed_s": round(elapsed, 1),
            "scaffold_tokens": scaffold_tokens,
            "scaffold_embedding": scaffold_embedding[:10] if scaffold_embedding else [],  # store first 10 dims (full in separate file)
            "response_embedding": response_embedding[:10] if response_embedding else [],
            "scaffold_response_cosine": cosine_similarity(scaffold_embedding, response_embedding) if scaffold_embedding and response_embedding else 0,
            **({f"probe_{k}": v for k, v in probe_result.items()} if probe_result else {}),
        }
        results.append(result)

        # Store full embeddings separately (too large for main JSONL)
        # We'll compute TE from these after the run

        label = "PROBE" if is_probe else "TURN"
        print(f"  [{topic_name[:8]:8s}|{condition[:8]:8s}|r{replication}] {label} {i:2d}/20 | {scaffold_tokens:4d} tok | {elapsed:.1f}s", end="")
        if is_probe:
            recall = probe_result.get("recall_rate", probe_result.get("resistance_score", "—"))
            print(f" | probe={recall}", end="")
        print()

    return results


# ── Proper Transfer Entropy on embeddings ────────────────────────────────────

def compute_te_from_embeddings(results: list[dict]) -> dict:
    """Compute transfer entropy using scaffold embedding similarities, not scalar entropy.

    TE(scaffold → response) = H(response_t | response_{t-1}) - H(response_t | response_{t-1}, scaffold_t)

    Uses cosine similarity bins instead of entropy scalars.
    """
    from pyitlib import discrete_random_variable as drv

    if len(results) < 4:
        return {"te": 0, "note": "too few turns"}

    # Get full embeddings for this session
    scaffold_embs = []
    response_embs = []

    for r in results:
        s_text = r.get("scaffold_snapshot", r.get("response", ""))  # fallback
        r_text = r.get("response", "")
        scaffold_embs.append(embed_text(s_text[:2000]))
        response_embs.append(embed_text(r_text[:2000]))

    # Compute pairwise cosine similarities for discretization
    # scaffold(t) vs response(t) similarity
    sr_sims = []
    for i in range(len(results)):
        if scaffold_embs[i] and response_embs[i]:
            sr_sims.append(cosine_similarity(scaffold_embs[i], response_embs[i]))
        else:
            sr_sims.append(0)

    # scaffold(t-1) vs scaffold(t) similarity (scaffold drift)
    ss_sims = [0]  # no previous for first turn
    for i in range(1, len(results)):
        if scaffold_embs[i-1] and scaffold_embs[i]:
            ss_sims.append(cosine_similarity(scaffold_embs[i-1], scaffold_embs[i]))
        else:
            ss_sims.append(0)

    # response(t-1) vs response(t) similarity (response drift)
    rr_sims = [0]
    for i in range(1, len(results)):
        if response_embs[i-1] and response_embs[i]:
            rr_sims.append(cosine_similarity(response_embs[i-1], response_embs[i]))
        else:
            rr_sims.append(0)

    # Discretize into bins for TE calculation
    n_bins = 4

    def discretize(values):
        arr = np.array(values[1:])  # skip first (no previous)
        if arr.std() < 0.001:
            return np.zeros(len(arr), dtype=int)
        edges = np.linspace(arr.min() - 0.001, arr.max() + 0.001, n_bins + 1)
        return np.digitize(arr, edges[:-1]) - 1

    sr_bins = discretize(sr_sims)  # scaffold-response coupling at t
    ss_prev_bins = discretize(ss_sims)  # scaffold drift from t-1 to t
    rr_bins = discretize(rr_sims)  # response drift from t-1 to t

    # TE = H(response_t | response_{t-1}) - H(response_t | response_{t-1}, scaffold_t)
    h_r_given_rprev = drv.entropy_conditional(sr_bins, rr_bins)
    joint_cond = rr_bins * 10 + ss_prev_bins
    h_r_given_rprev_s = drv.entropy_conditional(sr_bins, joint_cond)
    te = float(h_r_given_rprev - h_r_given_rprev_s)

    return {
        "te": round(te, 4),
        "mean_scaffold_response_cosine": round(float(np.mean(sr_sims)), 4),
        "mean_scaffold_drift": round(float(np.mean(ss_sims[1:])), 4),
        "mean_response_drift": round(float(np.mean(rr_sims[1:])), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    Path("results").mkdir(exist_ok=True)

    conditions = ["naked", "compressed", "clamped", "baseline"]
    n_replications = 5

    total_sessions = len(TOPICS) * len(conditions) * n_replications

    print("=" * 70)
    print("LPCI RIGOROUS VALIDATION")
    print(f"Topics: {list(TOPICS.keys())}")
    print(f"Conditions: {conditions}")
    print(f"Replications: {n_replications}")
    print(f"Total sessions: {total_sessions} × 20 turns = {total_sessions * 20} turns")
    print("=" * 70)

    all_results = []
    session_summaries = []
    session_count = 0

    for topic_name, topic_data in TOPICS.items():
        for condition in conditions:
            for rep in range(1, n_replications + 1):
                session_count += 1
                print(f"\n--- Session {session_count}/{total_sessions}: {topic_name} | {condition} | rep {rep} ---")

                try:
                    results = run_session(
                        topic_name=topic_name,
                        topic_data=topic_data,
                        condition=condition,
                        replication=rep,
                    )
                    all_results.extend(results)

                    # Compute TE for this session
                    te_result = compute_te_from_embeddings(results)

                    # Probe results
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

                # Save incrementally
                with open("results/lpci_rigorous.jsonl", "w") as f:
                    for r in all_results:
                        # Remove embeddings from main file (too large)
                        row = {k: v for k, v in r.items() if "embedding" not in k}
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                with open("results/lpci_rigorous_summary.jsonl", "w") as f:
                    for s in session_summaries:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ── Final analysis ────────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("FINAL ANALYSIS")
    print(f"{'=' * 70}")

    from scipy import stats as sp_stats

    # Group by condition
    by_condition = {}
    for s in session_summaries:
        c = s["condition"]
        if c not in by_condition:
            by_condition[c] = []
        by_condition[c].append(s)

    print(f"\n{'Condition':>12s}  {'TE mean':>8s}  {'TE std':>7s}  {'Recall':>7s}  {'Resist':>7s}  {'FC%':>5s}  {'Tokens':>7s}  {'S-drift':>8s}")
    for c in conditions:
        data = by_condition.get(c, [])
        if not data:
            continue
        tes = [d["te"] for d in data]
        recalls = [d["mean_recall"] for d in data if d["mean_recall"] is not None]
        resists = [d["mean_resistance"] for d in data if d["mean_resistance"] is not None]
        fcs = [d["false_claim_caught"] for d in data if d["false_claim_caught"] is not None]
        tokens = [d["final_scaffold_tokens"] for d in data]
        drifts = [d.get("mean_scaffold_drift", 0) for d in data]

        fc_pct = sum(1 for f in fcs if f) / max(len(fcs), 1) * 100

        print(f"{c:>12s}  {np.mean(tes):8.4f}  {np.std(tes):7.4f}  {np.mean(recalls):7.3f}  {np.mean(resists):7.3f}  {fc_pct:5.0f}%  {np.mean(tokens):7.0f}  {np.mean(drifts):8.4f}")

    # Significance tests
    print(f"\n## Pairwise Mann-Whitney: TE")
    for c in ["compressed", "clamped", "baseline"]:
        if c in by_condition and "naked" in by_condition:
            naked_te = [d["te"] for d in by_condition["naked"]]
            other_te = [d["te"] for d in by_condition[c]]
            if len(naked_te) >= 3 and len(other_te) >= 3:
                u, p = sp_stats.mannwhitneyu(naked_te, other_te, alternative="two-sided")
                print(f"  naked vs {c:12s}: U={u:.0f}  p={p:.4f}  sig={'*' if p < 0.05 else 'ns'}")

    print(f"\n## Pairwise Mann-Whitney: Recall")
    for c in ["compressed", "clamped", "baseline"]:
        if c in by_condition and "naked" in by_condition:
            naked_r = [d["mean_recall"] for d in by_condition["naked"] if d["mean_recall"] is not None]
            other_r = [d["mean_recall"] for d in by_condition[c] if d["mean_recall"] is not None]
            if len(naked_r) >= 3 and len(other_r) >= 3:
                u, p = sp_stats.mannwhitneyu(naked_r, other_r, alternative="two-sided")
                print(f"  naked vs {c:12s}: U={u:.0f}  p={p:.4f}  sig={'*' if p < 0.05 else 'ns'}")

    print(f"\nResults saved to:")
    print(f"  results/lpci_rigorous.jsonl ({len(all_results)} rows)")
    print(f"  results/lpci_rigorous_summary.jsonl ({len(session_summaries)} session summaries)")


if __name__ == "__main__":
    main()
