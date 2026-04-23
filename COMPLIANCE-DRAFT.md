# COMPLIANCE-DRAFT.md — langquant

**Status:** DRAFT. Internal self-audit artifact generated 2026-04-17.
Not a conformity declaration.

**System:** langquant — the LPCI SDK. Provides scaffold-monad (conditional-
state scaffolds for stateless LLMs), driftwatch (scaffold-drift detection),
and computetrace (provenance for scaffolded calls). 94 tests.

**Risk classification under EU AI Act:** The SDK itself is not a high-risk
AI system. It is developer infrastructure. Systems BUILT with langquant may
be high-risk if deployed in Annex III contexts (healthcare, employment,
credit, law enforcement). This document describes the SDK's compliance
posture and what a deployer must additionally do to make their downstream
system conformant.

---

## Article 9 — Risk management & data provenance

**Data sources used BY langquant itself:**
- None. langquant is a Python library that wraps and instruments caller
  LLM calls. It does not fetch or store data.
- No training data. langquant is deterministic over its inputs plus the
  caller's LLM provider.
- LLM provider is caller-supplied (OpenAI, Anthropic, local Ollama). Data
  sent to that provider is the caller's responsibility under Article 10.

**Risks identified:**
1. **Scaffold-drift unobserved.** If the caller does not enable driftwatch,
   scaffold compliance is not guaranteed across turns. Mitigation: driftwatch
   is enabled by default in scaffold-monad v0.2+.
2. **Prompt-injection via scaffold payload.** A malicious payload can subvert
   the scaffold's intended state. Mitigation: computetrace records the full
   scaffold+payload pair for every call, enabling post-hoc review.
3. **Provider outage.** langquant cannot guarantee availability of the
   wrapped LLM provider. Caller must implement fallback or degradation.

**Residual risk:** acceptable for experimentation and for wrapping audited
downstream systems. NOT acceptable as the sole control layer for high-risk
AI under Annex III — the deployer must add their own behavioral testing and
human oversight on top.

## Article 10 — Data governance

- No PII or caller data is stored by langquant. computetrace logs are
  caller-controlled (in-memory or written to a path the caller specifies).
- Caller is responsible for the lawful basis, retention, and access control
  of both their prompts and the LLM provider outputs.

## Article 14 — Human oversight & override

- **Override:** the caller invokes langquant; stopping is a normal Python
  `raise`. No autonomous control.
- **Human-in-the-loop hooks:** scaffold-monad supports a `checkpoint_fn`
  callback that can pause execution for human review. Recommended pattern
  for any Annex-III deployment built on langquant.
- **Audit trail:** computetrace produces a structured per-call record
  (scaffold, payload, provider, tokens, response hash) that is suitable
  for Article 12 record-keeping requirements.

## Article 15 — Accuracy, robustness, cybersecurity

- **Accuracy:** 94/94 tests green. Test suite covers scaffold-monad
  correctness, driftwatch sensitivity, computetrace completeness.
- **Robustness:** fuzz harness not yet implemented. Flagged as gap.
- **Cybersecurity:** langquant does not open network sockets; all network
  activity is via the wrapped provider SDK.

## Article 86 — Right to explanation

computetrace records (scaffold, payload, provider, response hash) provide
the raw material a downstream system needs to explain *why a given output
was produced*. langquant itself does not produce user-facing explanations —
it gives the deployer the inputs needed to produce one.

## Known limitations (disclosure)

1. No robustness fuzz harness yet (flagged).
2. No formal model registry integration; deployer must log provider/model
   versions themselves.
3. Scaffold efficacy depends on downstream model. LPCI held on qwen3.5 and
   claude families; other families untested at scale.
4. No out-of-the-box bias testing. Caller must pair with a bias evaluator.

## Remediation plan

- Add robustness fuzz harness (sprint 1)
- Publish the "deployer's compliance supplement" doc template (sprint 2)
- Add optional computetrace sink that writes to an append-only log suitable
  for Article 12 (sprint 2)
