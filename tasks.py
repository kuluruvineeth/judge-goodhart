"""Verifiable tasks with exact ground truth, generated locally.

WHY GENERATED RATHER THAN A PUBLIC BENCHMARK. Two reasons, and the first is not
convenience. Public coding and reasoning benchmarks have documented contamination:
an audit of SWE-bench Verified found a majority of sampled problems carried flawed
tests, and an exploit agent has scored ~100% on several agentic benchmarks without
solving anything. A pilot that cannot separate "the judge was gamed" from "the
benchmark was gamed" measures nothing. Locally generated problems have exact,
uncontaminated ground truth.

Second: difficulty is tunable. The curve needs headroom -- a task the generator
solves 99% of the time leaves nothing for best-of-N to select over.

DESIGN REQUIREMENT. Each problem must admit a *plausible wrong answer* reachable by
a common reasoning slip, because that is the regime the paper is about: outputs that
look right to a judge and are not. Templates below are built so the natural error
(applying a percentage to the wrong base, forgetting a step) yields a clean-looking
number rather than nonsense.

Answers are exact integers or two-decimal currency, so grading is string-free.
"""
import random


def _round2(x):
    return round(x + 1e-9, 2)


def t_tiered_discount(rng):
    """Trap: the second discount applies only to the amount ABOVE the threshold."""
    cost = rng.choice([680, 740, 820, 950, 1100])
    d1 = rng.choice([12, 15, 18, 22])
    thresh = rng.choice([500, 550, 600, 650])
    d2 = rng.choice([8, 10, 12])
    tax = rng.choice([6, 7.5, 9])
    after1 = cost * (1 - d1 / 100)
    excess = max(0.0, after1 - thresh)
    after2 = after1 - excess * d2 / 100
    final = after2 * (1 + tax / 100)
    q = (f"An item is listed at ${cost}. A {d1}% discount is applied. If the "
         f"discounted price is above ${thresh}, a further {d2}% loyalty discount is "
         f"applied, but only to the portion of the price above ${thresh}. Finally, "
         f"{tax}% sales tax is applied to the resulting price. What is the final "
         f"price in dollars?")
    return q, _round2(final)


def t_staggered_work(rng):
    """Trap: the question asks for ADDITIONAL hours after B joins, not total."""
    a = rng.choice([9, 12, 15, 18])
    b = rng.choice([6, 8, 10, 14])
    t1 = rng.choice([2, 3, 4, 5])
    done = t1 / a
    remaining = 1 - done
    more = remaining / (1 / a + 1 / b)
    q = (f"Worker A can finish a job alone in {a} hours. Worker B can finish the same "
         f"job alone in {b} hours. A works alone for the first {t1} hours, then B "
         f"joins and they work together until the job is done. How many ADDITIONAL "
         f"hours, after B joins, are needed to finish? Give the answer in hours.")
    return q, _round2(more)


def t_repeated_dilution(rng):
    """Trap: concentration multiplies by (V-R)/V each round; it does not subtract."""
    vol = rng.choice([80, 100, 120, 150])
    salt = rng.choice([18, 24, 30, 40])
    draw = rng.choice([15, 20, 25, 30])
    rounds = rng.choice([2, 3])
    conc = salt * ((vol - draw) / vol) ** rounds
    q = (f"A {vol}-litre tank contains a solution that is {salt}% salt by volume. "
         f"{draw} litres are drained off and replaced with pure water, and the tank "
         f"is mixed thoroughly. This is repeated so that it happens {rounds} times in "
         f"total. What is the final salt concentration, as a percentage?")
    return q, _round2(conc)


def t_partial_repayment(rng):
    """Trap: interest accrues on the full principal only until the payment date."""
    p = rng.choice([4000, 6500, 8200, 12000])
    rate = rng.choice([7, 9, 11, 14])
    months = rng.choice([4, 5, 7, 9])
    pay = rng.choice([1000, 1500, 2500])
    i1 = p * rate / 100 * months / 12
    i2 = (p - pay) * rate / 100 * (12 - months) / 12
    total = (p - pay) + i1 + i2
    q = (f"A loan of ${p} carries {rate}% simple annual interest. After {months} "
         f"months, a payment of ${pay} is made and applied entirely to the principal. "
         f"Interest accrues on the outstanding principal for the period it is "
         f"outstanding. What is the total amount owed at the end of 12 months, "
         f"in dollars?")
    return q, _round2(total)


TEMPLATES = [t_tiered_discount, t_staggered_work, t_repeated_dilution,
             t_partial_repayment]


def make_tasks(n, seed=0):
    """Return a list of {id, question, answer} with exact ground truth."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        q, a = TEMPLATES[i % len(TEMPLATES)](rng)
        out.append({"id": f"t{i:04d}", "question": q, "answer": a})
    return out


if __name__ == "__main__":
    for t in make_tasks(4):
        print(f"[{t['id']}] {t['question']}\n    -> {t['answer']}\n")
