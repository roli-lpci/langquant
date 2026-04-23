#!/usr/bin/env python3
"""
LangQuant — Language State Compression Experiment Runner
Hermes Labs, 2026

Measures whether structured language states (scaffolds) make small models
produce behavior equivalent to larger models on naive prompts.

Core metric: compression_ratio = behavioral_complexity(output) / entropy(scaffold)

Reuses patterns from:
  - epistemic-experiments/experiment_runner.py (Ollama calls, JSON extraction)
  - quickthink/run_suite.py (locking, manifests, JSONL streaming, resume)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
import urllib.request
import urllib.error
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    "qwen3.5:27b",
]

# Scaffold conditions: naked → progressively richer language states
SCAFFOLD_CONDITIONS = {
    "naked": None,  # Raw prompt, no scaffold
    "contrastive": "contrastive_markers",
    "quickthink": "quickthink_grammar",
    "full_scaffold": "full_scaffold",
    "hypothesis_artifacts": "hypothesis_artifacts",
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    task_id: str
    group: str           # e.g. "reasoning", "classification", "calibration"
    prompt: str          # The raw task prompt (before scaffolding)
    expected: str        # Expected answer or rubric
    difficulty: str      # "easy", "medium", "hard"


@dataclass
class ScaffoldState:
    """A language state applied to a task prompt."""
    condition: str       # Key from SCAFFOLD_CONDITIONS
    prefix: str = ""     # Text prepended to prompt
    suffix: str = ""     # Text appended to prompt
    token_count: int = 0 # Tokens in scaffold (prefix + suffix)

    def apply(self, prompt: str) -> str:
        parts = [p for p in [self.prefix, prompt, self.suffix] if p]
        return "\n\n".join(parts)


@dataclass
class TrialResult:
    timestamp: str
    model: str
    task_id: str
    group: str
    condition: str
    run_index: int
    # Raw
    raw_response: str
    response_length: int
    latency_ms: float
    parse_success: bool
    # Behavioral complexity metrics
    bc_word_count: int
    bc_unique_words: int
    bc_vocabulary_richness: float   # unique/total
    bc_sentence_count: int
    bc_avg_sentence_length: float
    bc_reasoning_signals: int       # "because", "therefore", "however", etc.
    bc_hedging_count: int
    bc_structural_markers: int      # lists, enumerations, comparisons
    bc_composite_score: float       # Weighted composite
    # Information-theoretic metrics on scaffold
    scaffold_token_count: int
    scaffold_entropy: float         # Shannon entropy of scaffold token distribution
    # The novel metric
    compression_ratio: float        # bc_composite_score / scaffold_entropy (0 if naked)
    # Task performance
    task_score: float               # 0-1 correctness (rule-based or judge)
    task_notes: str


# ── Ollama Integration (from experiment_runner.py) ───────────────────────────

def call_ollama(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    timeout: int = 180,
    ollama_url: str = OLLAMA_URL,
) -> tuple[str, float]:
    """Call Ollama chat API. Returns (response_text, latency_ms).
    Uses /api/chat with think=False to avoid Qwen3.5 thinking-only responses.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }).encode()

    req = urllib.request.Request(
        f"{ollama_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            latency = (time.monotonic() - t0) * 1000
            content = data.get("message", {}).get("content", "")
            # Strip any residual thinking tags
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content, latency
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return f"ERROR: {e}", latency


def extract_json(text: str) -> dict | None:
    """Extract JSON from model response, handling markdown fences and thinking tags."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:  # silent — fall through to next extraction strategy
            pass
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:  # silent — fall through to whole-text parse
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# ── Information-Theoretic Metrics ────────────────────────────────────────────

def tokenize_simple(text: str) -> list[str]:
    """Whitespace + punctuation tokenizer. Good enough for entropy measurement."""
    return re.findall(r'\b\w+\b', text.lower())


def shannon_entropy(tokens: list[str]) -> float:
    """Shannon entropy of token distribution in bits."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def token_count(text: str) -> int:
    """Approximate token count via whitespace split."""
    return len(text.split())


# ── Behavioral Complexity Metrics ────────────────────────────────────────────

REASONING_SIGNALS = [
    r'\bbecause\b', r'\btherefore\b', r'\bhowever\b', r'\balthough\b',
    r'\bconsequently\b', r'\bnevertheless\b', r'\bmoreover\b', r'\bfurthermore\b',
    r'\bimplies?\b', r'\bthus\b', r'\bhence\b', r'\bin contrast\b',
    r'\bon the other hand\b', r'\bif\s+.*\bthen\b', r'\bwhereas\b',
    r'\bdespite\b', r'\bsince\b', r'\bgiven that\b',
]

HEDGING_PATTERNS = [
    r'\bmay\b', r'\bmight\b', r'\bcould\b', r'\bpossibly\b', r'\bperhaps\b',
    r'\bunclear\b', r'\buncertain\b', r'\btentativ', r'\bpreliminar',
    r'\bwould need\b', r'\bwould require\b', r'\bdepends on\b',
    r'\bfurther research\b', r'\bmore studies\b', r'\bcaution\b',
    r'\bsuggest\b', r'\bimpl(?:y|ies)\b',
]

STRUCTURAL_MARKERS = [
    r'^\s*[\d]+[.)]\s',      # Numbered lists
    r'^\s*[-*]\s',            # Bullet points
    r'\bfirst(?:ly)?\b',      # Enumeration words
    r'\bsecond(?:ly)?\b',
    r'\bthird(?:ly)?\b',
    r'\bfinally\b',
    r'\bin summary\b',
    r'\bin conclusion\b',
    r'\bcompared to\b',
    r'\bversus\b',
    r'\bvs\.?\b',
]


def count_pattern_hits(text: str, patterns: list[str]) -> int:
    lower = text.lower()
    total = 0
    for p in patterns:
        total += len(re.findall(p, lower, re.MULTILINE))
    return total


def measure_behavioral_complexity(text: str) -> dict:
    """Measure multiple dimensions of behavioral complexity in a response."""
    words = tokenize_simple(text)
    word_count = len(words)
    unique_words = len(set(words))
    vocab_richness = unique_words / max(word_count, 1)

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_len = word_count / max(sentence_count, 1)

    reasoning = count_pattern_hits(text, REASONING_SIGNALS)
    hedging = count_pattern_hits(text, HEDGING_PATTERNS)
    structural = count_pattern_hits(text, STRUCTURAL_MARKERS)

    # Composite: weighted sum normalized to roughly 0-100 range
    # Weights emphasize reasoning depth and vocabulary over raw length
    composite = (
        vocab_richness * 20
        + min(reasoning, 15) * 3        # Cap at 15 to avoid runaway
        + min(structural, 10) * 2       # Cap at 10
        + min(sentence_count, 20) * 0.5 # Reward structure, cap at 20
        + min(hedging, 10) * 1          # Hedging = nuance signal
    )

    return {
        "bc_word_count": word_count,
        "bc_unique_words": unique_words,
        "bc_vocabulary_richness": round(vocab_richness, 4),
        "bc_sentence_count": sentence_count,
        "bc_avg_sentence_length": round(avg_sentence_len, 2),
        "bc_reasoning_signals": reasoning,
        "bc_hedging_count": hedging,
        "bc_structural_markers": structural,
        "bc_composite_score": round(composite, 2),
    }


# ── Scaffold Builders ────────────────────────────────────────────────────────

def build_scaffold(condition: str, task: Task) -> ScaffoldState:
    """Build a scaffold state for a given condition and task."""

    if condition == "naked":
        return ScaffoldState(condition="naked", token_count=0)

    elif condition == "contrastive":
        prefix = (
            f"Task type: {task.group}. "
            f"This is a {task.difficulty} {task.group} task. "
            f"Focus on the specific question asked. "
            f"This is NOT a creative writing task. "
            f"This is NOT a summarization task. "
            f"Provide a precise, well-reasoned answer."
        )
        return ScaffoldState(
            condition="contrastive",
            prefix=prefix,
            token_count=token_count(prefix),
        )

    elif condition == "quickthink":
        prefix = (
            "Before answering, create a compressed plan using this exact format:\n"
            "g:<goal in 3-5 words>;c:<key constraints>;s:<steps>;r:<core reasoning>\n\n"
            "Then provide your answer.\n"
        )
        return ScaffoldState(
            condition="quickthink",
            prefix=prefix,
            token_count=token_count(prefix),
        )

    elif condition == "full_scaffold":
        prefix = (
            f"Task type: {task.group} ({task.difficulty})\n"
            f"This is NOT a creative writing task. This is NOT a summarization task.\n"
            f"Focus on the specific question asked.\n\n"
            f"Before answering, create a compressed plan:\n"
            f"g:<goal>;c:<constraints>;s:<steps>;r:<reasoning>\n\n"
            f"Requirements:\n"
            f"- Be precise and specific\n"
            f"- Show your reasoning chain\n"
            f"- If uncertain, say so explicitly rather than guessing\n"
            f"- Address the question directly, do not pad with generic context\n"
        )
        return ScaffoldState(
            condition="full_scaffold",
            prefix=prefix,
            token_count=token_count(prefix),
        )

    elif condition == "hypothesis_artifacts":
        # Richest state: contrastive + quickthink + structured constraints + exemplar
        prefix = (
            f"## Task Context\n"
            f"Type: {task.group} | Difficulty: {task.difficulty}\n"
            f"This IS: a precise analytical task requiring careful reasoning.\n"
            f"This is NOT: creative writing, summarization, or opinion.\n\n"
            f"## Method\n"
            f"1. Compressed plan: g:<goal>;c:<constraints>;s:<steps>;r:<reasoning>\n"
            f"2. Identify what you know vs what you're uncertain about\n"
            f"3. State your answer with explicit confidence level\n\n"
            f"## Quality Constraints\n"
            f"- Every claim must have a reason\n"
            f"- Distinguish facts from inferences\n"
            f"- If the answer requires information you don't have, say so\n"
            f"- Prefer structured output (lists, steps) over prose\n"
            f"- No filler, no hedging-as-padding, no generic disclaimers\n"
        )
        return ScaffoldState(
            condition="hypothesis_artifacts",
            prefix=prefix,
            token_count=token_count(prefix),
        )

    else:
        raise ValueError(f"Unknown scaffold condition: {condition}")


# ── Task Scoring ─────────────────────────────────────────────────────────────

def score_task(task: Task, response: str) -> tuple[float, str]:
    """
    Score a response against the task's expected answer.
    Returns (score 0.0-1.0, notes).
    Simple keyword/pattern matching for now — extend with LLM judge later.
    """
    response_lower = response.lower()
    expected_lower = task.expected.lower()

    # Check for exact containment of expected answer
    if expected_lower in response_lower:
        return 1.0, "exact_match"

    # Check for key terms overlap
    expected_terms = set(tokenize_simple(expected_lower))
    response_terms = set(tokenize_simple(response_lower))
    if not expected_terms:
        return 0.5, "no_expected_terms"

    overlap = len(expected_terms & response_terms) / len(expected_terms)
    if overlap >= 0.8:
        return 0.9, f"high_overlap_{overlap:.2f}"
    elif overlap >= 0.5:
        return 0.6, f"partial_overlap_{overlap:.2f}"
    elif overlap >= 0.2:
        return 0.3, f"low_overlap_{overlap:.2f}"
    else:
        return 0.0, f"no_overlap_{overlap:.2f}"


# ── Locking & Manifests (from run_suite.py) ──────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SystemExit(
            f"Lock exists: {lock_path}. Another run active, or remove stale lock."
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "timestamp": now_iso()}) + "\n")


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:  # silent — idempotent: lock already released
        pass


def load_existing_offsets(path: Path) -> dict[tuple[str, str, str], int]:
    """Load existing run offsets for resume support."""
    offsets: dict[tuple[str, str, str], int] = {}
    if not path.exists():
        return offsets
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (
                str(row.get("model", "")),
                str(row.get("condition", "")),
                str(row.get("task_id", "")),
            )
            run_index = int(row.get("run_index", 0) or 0)
            if run_index > offsets.get(key, 0):
                offsets[key] = run_index
    return offsets


def write_manifest(
    manifest_path: Path,
    task_set_path: Path,
    models: list[str],
    conditions: list[str],
    runs: int,
    temperature: float,
    task_count: int,
) -> None:
    payload = {
        "experiment": "langquant",
        "timestamp": now_iso(),
        "git_sha": git_sha(),
        "task_set_path": str(task_set_path),
        "task_set_sha256": file_sha256(task_set_path),
        "models": models,
        "conditions": conditions,
        "runs_per_cell": runs,
        "temperature": temperature,
        "task_count": task_count,
        "total_trials": len(models) * len(conditions) * task_count * runs,
        "metrics": [
            "task_score", "bc_composite_score", "scaffold_entropy",
            "compression_ratio", "latency_ms",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ── Task Set ─────────────────────────────────────────────────────────────────

def load_task_set(path: Path) -> list[Task]:
    """Load tasks from JSONL file."""
    tasks = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks.append(Task(
                task_id=str(item["task_id"]),
                group=str(item["group"]),
                prompt=str(item["prompt"]),
                expected=str(item["expected"]),
                difficulty=str(item.get("difficulty", "medium")),
            ))
    return tasks


# ── Main Experiment Loop ─────────────────────────────────────────────────────

def run_trial(
    model: str,
    task: Task,
    condition: str,
    run_index: int,
    temperature: float,
    ollama_url: str,
) -> dict:
    """Run a single trial: scaffold + call + measure."""
    # Build scaffold
    scaffold = build_scaffold(condition, task)
    full_prompt = scaffold.apply(task.prompt)

    # Call model
    response, latency_ms = call_ollama(
        model=model,
        prompt=full_prompt,
        temperature=temperature,
        ollama_url=ollama_url,
    )

    # Measure behavioral complexity
    bc = measure_behavioral_complexity(response)

    # Measure scaffold entropy
    scaffold_text = f"{scaffold.prefix} {scaffold.suffix}".strip()
    scaffold_tokens = tokenize_simple(scaffold_text)
    s_entropy = shannon_entropy(scaffold_tokens)
    s_token_count = len(scaffold_tokens)

    # Score task performance
    task_score, task_notes = score_task(task, response)

    # Compute compression ratio
    if s_entropy > 0:
        compression_ratio = bc["bc_composite_score"] / s_entropy
    else:
        compression_ratio = 0.0  # Naked condition: no scaffold to compress through

    return {
        "timestamp": now_iso(),
        "model": model,
        "task_id": task.task_id,
        "group": task.group,
        "difficulty": task.difficulty,
        "condition": condition,
        "run_index": run_index,
        "scaffold_token_count": s_token_count,
        "scaffold_entropy": round(s_entropy, 4),
        "full_prompt_length": len(full_prompt),
        "raw_response": response[:3000],
        "response_length": len(response),
        "latency_ms": round(latency_ms, 2),
        "parse_success": not response.startswith("ERROR:"),
        "task_score": round(task_score, 2),
        "task_notes": task_notes,
        **bc,
        "compression_ratio": round(compression_ratio, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LangQuant — Language State Compression Experiment Runner"
    )
    parser.add_argument(
        "--task-set", default="tasks/task_set.jsonl",
        help="Path to JSONL task set",
    )
    parser.add_argument(
        "--out", default="results/run_results.jsonl",
        help="Path for JSONL output",
    )
    parser.add_argument(
        "--manifest-out", default="results/run_manifest.json",
        help="Path for reproducibility manifest",
    )
    parser.add_argument(
        "--models", nargs="*", default=MODELS,
    )
    parser.add_argument(
        "--conditions", nargs="*",
        default=list(SCAFFOLD_CONDITIONS.keys()),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    task_set_path = Path(args.task_set)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest_out)

    if not task_set_path.exists():
        raise SystemExit(f"Task set not found: {task_set_path}")

    tasks = load_task_set(task_set_path)
    if args.limit > 0:
        tasks = tasks[:args.limit]
    if not tasks:
        raise SystemExit("No tasks loaded")

    for c in args.conditions:
        if c not in SCAFFOLD_CONDITIONS:
            raise SystemExit(f"Unknown condition '{c}', expected one of {list(SCAFFOLD_CONDITIONS.keys())}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.append else "w"
    offsets = load_existing_offsets(out_path) if args.append else {}
    lock_path = Path(f"{out_path}.lock")
    acquire_lock(lock_path)

    try:
        total = len(args.models) * len(args.conditions) * len(tasks) * args.runs
        done = 0

        print(f"LangQuant experiment: {len(args.models)} models x {len(args.conditions)} conditions x {len(tasks)} tasks x {args.runs} runs = {total} trials")
        print(f"Output: {out_path}")

        with out_path.open(write_mode, encoding="utf-8") as fh:
            for model in args.models:
                for condition in args.conditions:
                    for task in tasks:
                        offset = offsets.get((model, condition, task.task_id), 0)
                        for run_index in range(1, args.runs + 1):
                            if run_index <= offset:
                                done += 1
                                continue

                            result = run_trial(
                                model=model,
                                task=task,
                                condition=condition,
                                run_index=run_index,
                                temperature=args.temperature,
                                ollama_url=args.ollama_url,
                            )
                            fh.write(json.dumps(result, ensure_ascii=True) + "\n")
                            fh.flush()
                            done += 1

                            if done % 10 == 0 or done == total:
                                score = result["task_score"]
                                bc = result["bc_composite_score"]
                                cr = result["compression_ratio"]
                                print(
                                    f"[{done}/{total}] {model} | {condition} | "
                                    f"{task.task_id} | score={score} bc={bc} cr={cr}"
                                )

        write_manifest(
            manifest_path=manifest_path,
            task_set_path=task_set_path,
            models=list(args.models),
            conditions=list(args.conditions),
            runs=args.runs,
            temperature=args.temperature,
            task_count=len(tasks),
        )
        print(f"\nDone. Results: {out_path}")
        print(f"Manifest: {manifest_path}")
        return 0

    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
