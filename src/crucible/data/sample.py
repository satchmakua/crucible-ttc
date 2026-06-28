"""A tiny bundled math dataset + scripted model outputs for the mock backend.

Six GSM8K-style problems with gold answers, paired with canned "model" completions —
four correct, two wrong — so a `pass1` run on the mock backend yields a deterministic,
*meaningful* pass@1 of 4/6. The set deliberately exercises every load-bearing path:
``\boxed{}`` extraction, "the answer is …" phrasing, wrong answers, and symbolic
equivalence (``1/4`` ≡ ``0.25``).
"""

from __future__ import annotations

from crucible.domain.types import Problem

SAMPLE_PROBLEMS: tuple[Problem, ...] = (
    Problem(
        id="1",
        prompt=(
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did she sell altogether?"
        ),
        answer="72",
        difficulty="easy",
    ),
    Problem(id="2", prompt="What is 12 multiplied by 12?", answer="144", difficulty="easy"),
    Problem(
        id="3",
        prompt="A baker had 30 cupcakes and sold 18 of them. How many are left?",
        answer="12",
        difficulty="easy",
    ),
    Problem(
        id="4",
        prompt="A train travels 60 miles in 1.5 hours. What is its speed in mph?",
        answer="40",
        difficulty="medium",
    ),
    Problem(id="5", prompt="Compute 1/2 + 1/2.", answer="1", difficulty="easy"),
    Problem(
        id="6",
        prompt="Write one quarter (1/4) as a decimal.",
        answer="0.25",
        difficulty="medium",
    ),
)

# Canned "model" completions keyed by problem id. Correct: 1, 4, 5, 6. Wrong: 2, 3.
SAMPLE_SCRIPTS: dict[str, list[str]] = {
    "1": [
        "Natalia sold 48 clips in April.\n\n"
        "In May she sold half as many: 48 / 2 = 24.\n\n"
        "Altogether: 48 + 24 = 72. The answer is \\boxed{72}."
    ],
    "2": [
        "I need 12 times 12.\n\n"
        "Multiplying: 12 * 12 = 124.\n\n"  # deliberately wrong
        "So the answer is \\boxed{124}."
    ],
    "3": [
        "The baker started with 30 cupcakes and sold 18.\n\n"
        "30 + 18 = 48.\n\n"  # deliberately wrong (added instead of subtracted)
        "\\boxed{48}"
    ],
    "4": [
        "Speed is distance divided by time.\n\n"
        "60 miles / 1.5 hours = 40.\n\n"
        "The answer is \\boxed{40} mph."
    ],
    "5": [
        "Adding the two halves: 1/2 + 1/2 = 2/2 = 1.\n\n\\boxed{1}"
    ],
    "6": [
        "One quarter as a decimal.\n\n"
        "I'll leave it as the fraction 1/4 rather than converting.\n\n"
        "\\boxed{1/4}"  # equivalent to gold 0.25 — tests symbolic equivalence
    ],
}
