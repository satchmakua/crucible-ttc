# 2. Ports-and-adapters core with first-class compute accounting

- **Status:** Accepted
- **Date:** 2026-06-27

## Context

Crucible compares several test-time search strategies (majority@N, best-of-N,
PRM-weighted best-of-N, beam/DVTS, MCTS) across swappable inference backends
(Ollama, vLLM, hosted), two verifier families (programmatic + learned PRM), and
multiple datasets (GSM8K, MATH-500, HumanEval, MBPP). Two failure modes loom:

1. **Coupling.** If a search strategy reaches directly into an inference backend or
   a dataset format, every new backend/verifier/dataset becomes a rewrite, and the
   core is impossible to unit-test without a live model.
2. **Dishonest measurement.** The headline artifact is an *accuracy-vs-compute*
   curve. A PRM-guided search that calls a 7B verifier many times per problem is not
   comparable to best-of-N at the same N unless the verifier's compute is counted.
   This is the single most common way hobby reproductions mislead themselves.

## Decision

- **Hexagonal / ports-and-adapters.** The search/verification **core** depends only
  on a small set of `Protocol`s (`PolicyModel`, `OutcomeVerifier`,
  `ProcessVerifier`, `SearchStrategy`) and value types (`Problem`, `Step`, `Trace`,
  `Compute`). Backends, verifiers, datasets, and reporting are **adapters** behind
  those ports. A new backend is a new adapter, never a change to the search core.
- **Compute accounting is a value type, not an afterthought.** Every `Trace` carries
  a frozen `Compute` (policy generated tokens, policy forward calls, verifier
  forward calls, wall seconds). Adapters populate it; the runner sums it; reports
  plot accuracy against total tokens (policy + verifier). Verifier calls are
  **counted, never free.**

## Consequences

- The core is testable with a `ScriptedPolicy` and a `MockVerifier` — no GPU, no
  network — which is what makes M0 run and the build loop fast.
- Adding Ollama/vLLM/PRM/MCTS is additive; the strategy code is untouched.
- A small tax: every adapter must honestly report its token/call counts, and the
  `Compute` type must be threaded through. This is deliberate — it is the honesty
  layer the whole project rests on.
- See `DESIGN.md` §4–§6 for the full protocol definitions and rationale.
