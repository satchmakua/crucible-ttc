# PROGRESS — Crucible

A build log of what shipped and the notable decisions behind it. **Keep it honest** —
this is the working memory between build sessions. The forward-looking plan and
acceptance tests live in [ROADMAP.md](ROADMAP.md); this is the backward-looking
"what got done and why" companion.

**Current phase:** Phase 0 complete (M0). Next up: **M1** (Ollama + GSM8K pass@1).

## State of the tree

| Component | Where | Status |
|---|---|---|
| Value types + ports (the hexagon) | `domain/types.py`, `domain/ports.py` | ✅ M0 |
| Compute accounting | `domain/types.py::Compute` | ✅ M0 |
| Config (YAML ↔ dataclass) | `config.py` | ✅ M0 |
| Step segmentation + token approx | `segment.py` | ✅ M0 (used fully in M4) |
| Wilson CIs | `stats.py` | ✅ M0 |
| Mock policy (ScriptedPolicy) | `inference/mock.py` | ✅ M0 |
| Ollama policy | `inference/ollama.py` | 🟡 written, wired in M1 |
| Answer extraction | `verify/answer_extract.py` | ✅ M0 |
| Math outcome verifier | `verify/math_outcome.py` | ✅ M0 |
| PRM (process verifier) | `verify/` | ⬜ M3 |
| Code-execution verifier | `verify/` | ⬜ M5 |
| pass1 strategy + registry | `search/` | ✅ M0 |
| best_of_n / beam / mcts | `search/` | ⬜ M2 / M4 / M6 |
| Sample dataset + registry | `data/` | ✅ M0 |
| HF dataset loaders | `data/` | ⬜ M1 (GSM8K/MATH-500), M5 (code) |
| Experiment runner | `runner.py` | ✅ M0 |
| Run records (JSON/CSV) + summary | `report.py` | ✅ M0 |
| Accuracy-vs-compute curve | `report.py` | ⬜ M2 |
| CLI (run/report/sweep/version) | `cli.py` | ✅ M0 (sweep stubbed → M2) |

---

## M0 — Skeleton & it runs · built 2026-06-27 · ✓ verified at scaffold

The full vertical spine, end-to-end, with **zero external dependencies** — a fresh
session can `pip install -e ".[dev]"` and immediately run and test it.

**What shipped:**
- Hexagonal core: `domain/types.py` (`Problem`, `Step`, `Trace`, `Compute`, `Result`,
  `Verdict`) and `domain/ports.py` (the `PolicyModel` / `OutcomeVerifier` /
  `ProcessVerifier` / `SearchStrategy` protocols). The search core depends only on
  these.
- **Compute accounting** as a first-class value type (`Compute`, additive, with
  `total_tokens` = policy + verifier) — the honesty layer from day one.
- A deterministic **`ScriptedPolicy`** mock backend, plus a real **`OllamaPolicy`**
  (httpx) written and ready to wire in M1.
- Math **answer extraction** (`\boxed{}`, "answer is …", number fallback) +
  **`math-verify`** outcome verifier with a SymPy/string fallback.
- A bundled **`sample`** dataset (6 GSM8K-style problems + canned outputs: 4 right, 2
  wrong) that exercises every path, including symbolic equivalence (`1/4 ≡ 0.25`).
- `runner.py` (problem × method → `Result`), `report.py` (JSON/CSV records + a rich
  console summary with **Wilson CIs**), and a Typer **CLI** (`run`/`report`/`version`;
  `sweep` stubbed with an M2 message).
- Project hygiene: `pyproject.toml` (light base deps; `datasets`/`prm`/`vllm`/`dev`
  extras), ruff + mypy(strict) config, GitHub Actions CI, `.env.example`, two ADRs.

**How it was verified (concrete evidence):**
- `ruff check .` → clean; `mypy src` → no issues in 23 files; `pytest` → **33 passed**.
- `python -m crucible run --method pass1 --dataset sample --policy mock` →
  **66.7% (4/6)** pass@1, Wilson CI [30.0%, 90.3%], 105 total tokens, 6 verifier calls;
  problem 6 (`\boxed{1/4}` vs gold `0.25`) correctly marked correct by symbolic
  equivalence; a record was written under `runs/`.
- `crucible report <run>` and `crucible run --config configs/sample.yaml` both work;
  `--method mcts` returns the milestone-aware error (exit 1).

**Gotchas for future sessions:**
- `math-verify` 0.9.0's built-in timeout uses signals/subprocesses that are fragile on
  Windows (and crash under `python -c`). We **disable** the timeout (`parsing_timeout=
  None`, `timeout_seconds=None`) and silence its warning logger in `math_outcome.py`.
  Inputs are short, so this is safe; revisit if very long expressions appear.
- Editable install must happen **after** `src/crucible/` exists, or the package path
  won't register (hit this once; `pip install -e .` again fixed it).
- Console output is kept ASCII (Windows cp1252 can't encode `✓`/`·`); `cli.main()` also
  reconfigures stdout to UTF-8 defensively.
- `runs/` is gitignored — run records are reproducible from `config.json`.
