# Well Calibrated, Wrongly Ordered

Optimising against an LLM judge buys confident fabrication — but only where the judge's
ranking disagrees with the truth.

[Read the paper](main.pdf)

## The argument

LLM judges are used to *select*: best-of-N, rejection sampling, iterative revision,
preference data. Every one of those is an optimiser pointed at the judge, and the folk
expectation is that a judge validated once on ordinary outputs stops describing reality once
outputs are chosen to please it.

We tested that on a domain with exact, freely derived ground truth. **The expectation is
wrong in the regime most teams operate in, and severe in a narrower one that is easy to
miss.**

![Fabrication on problems that cannot be answered. Both arms start from the same solution
with the same revision budget; only the instruction differs.](figures/fig1_fabrication.png)

Three experiments:

| condition | pressure | judge vs truth | result |
|---|---|---|---|
| 1 | best-of-24, i.i.d. | aligned | **no effect** — accuracy rises +16.45 pp (5.6 SE) |
| 2 | 5 rounds optimising the score | aligned | **no effect** — score +11.5, accuracy rose with it |
| 3 | 5 rounds optimising the score | **opposed** | **+75.0 pp fabrication (13.3 SE)** |

Experiment 3 constructs the opposing condition rather than hoping to meet it: half the
problems have one required quantity deleted, so the correct answer is a refusal and *any*
number is a fabrication. The generator is explicitly told refusal is legitimate, so
fabrication measures the optimiser and not the prompt.

The two nulls are not failures. They are the boundary that gives the third result meaning:
**the danger is not optimisation pressure, and it is not best-of-N. It is optimisation
pressure applied where the judge's ordering diverges from the truth.**

## The mechanism, and the exchange rate

The optimiser is not misreading its instructions — it is climbing a real gradient. On
unanswerable problems the judge scores fabricated answers **17.0** against **10.5** for
honest refusals: **+6.5 points at 9.3 SE**. On the 65 within-task transitions where the
optimised arm flipped from refusing to fabricating, the score rose **+5.8 points** (SE 1.3).

Approximately six points of judge score, purchased with **75 percentage points of
correctness**.

## Well calibrated, wrongly ordered

The judge is **not deceived**. It scores those fabrications 10–17 out of 100 — by any
calibration measure it knows they are bad. What is broken is the **ranking**: it places
confident fabrication above honest refusal. An optimiser consumes only the ordering.

So mean agreement is the wrong summary statistic for a judge that will be optimised against.
The practical check is not "how often does my judge agree with humans" but **"how does my
judge rank cases where the honest answer is unattractive"** — refusals, admissions of
insufficient information, hedged answers, anything correct but unimpressive.

## A secondary result: judge choice is a measurement, not a default

Measured on 60 solutions, balanced 30 correct / 30 incorrect, identical prompt:

| judge | mean | AUC | >90 | headroom |
|---|---|---|---|---|
| `claude-haiku-4-5` | 91.7 | 0.644 | 95% | 8.3 |
| `claude-sonnet-5` | 67.0 | **0.851** | 63% | 33.0 |
| `claude-opus-5` | 69.8 | 0.738 | 67% | 30.2 |

The small model sits on the ceiling — selection against it is close to random. **The largest,
most expensive model discriminates worse than the mid-tier one at roughly five times the
price.** AUC, a pure ranking measure, separates these judges where mean score does not —
which is the paper's thesis appearing a second time.

## Reproducing

```
python3 power.py           # pre-spend power analysis; mutants must fail
python3 run_pilot.py       # experiment 1
python3 optimize.py        # experiment 2 (two-arm)
python3 optimize2.py       # experiment 3 (two-arm, unanswerable half)
python3 analyze*.py        # curves and paired statistics
python3 figures.py         # figures, with generation-time guards
tectonic main.tex          # the paper
```

Requires `ANTHROPIC_API_KEY`. Total API spend across all three experiments was roughly $80.

## What is in here that usually is not published

- `runs/*.jsonl` — **every raw generation and judge score**, so any number in the paper can be
  re-derived rather than taken on trust.
- `power.py` — the power analysis run **before** spending, which falsified the original thesis
  on its first execution and forced the sharper conditional claim that survived.
- `CLAIMS.md` — the claim sheet, including the **must-not-claim list**: results that looked
  real and did not survive testing. The efficiency decay in experiment 1 reads as 81.5% → 68.5%
  in aggregate and is 0.8 SE once paired per task. It is recorded rather than quietly dropped.
- Mutation tests in every analysis script: shuffled judge must flatten the curve, a judge given
  ground truth must reach the oracle, an inverted judge must make things worse. A check with no
  demonstrated failure mode is decoration.

## Limitations

One task family, one generator, one judge, 60 unanswerable problems at four rounds. The
unanswerable construction is synthetic. Two experiments were **not** run and are named in the
paper: a judge explicitly prompted to reward appropriate refusal — the first fix a
practitioner would try, which might eliminate the effect entirely — and a subjective domain,
where ground truth is unavailable and the coincidence producing our nulls may break for
unrelated reasons.

## Licence

See [LICENSE](LICENSE).
