# Paper 3 — verified claim sheet

Working title: **The Goodhart Curve for LLM Judges**
Subtitle candidate: *Judge validity as a function of optimization pressure*

Status: thesis selected 2026-08-03 after the previous candidate (endpoint noise floor) was killed by
prior art. Nothing below is written until it is checked. Claims verified by fetching the primary
source are marked **[V]**; claims taken from a scout and not yet re-extracted are marked `[unverified]`
and **may not be written into the paper**.

---

## LAW 0 — THE MATTER PHASE (answered before any DOI was checked)

**1. Who is hurt.** An ML engineer running best-of-16 with an LLM judge over a support summariser.
Judge score moves 7.1 → 8.4. They validated that judge months ago at ~81% agreement with humans — on
*unoptimized single samples*. But best-of-16 is a search for the judge's blind spots, so at N=16 the
real agreement may be far lower and most of the "gain" is the selector finding judge-pleasing
artifacts. Dashboard green, user outcomes flat. **There is no published number they can look up.**

**2. What they do differently.** Report judge agreement *at their deployed optimization pressure*
rather than as a one-off scalar; read off the N at which their judge crosses their tolerance and cap
best-of-N there; and select judges by the *flatness* of the curve rather than by mean agreement —
which can pick a different judge than the one they would have chosen.

**3. How big, in their units.** The endpoint of the curve is measured and catastrophic (see [V1]).
The interior — where every team actually operates — is unmeasured. That gap is the paper.

**4. Why still unsolved.** It sits in a seam. RLHF theorists own overoptimization but only for
*reward models* graded by a synthetic gold RM [V2]. The eval community owns judges but measures them
*statically*, on i.i.d. unoptimized outputs. Nobody has crossed it.

---

## THE CLAIM I INTEND TO DEFEND — **revised 2026-08-03 by `power.py`, before any API spend**

The first version of this section read: *"Judge–ground-truth agreement decays as a function of
optimization pressure."* **The simulation falsified that as a general claim on the first run.** With
a judge-pleasing feature that is merely *uninformative* (verbosity, formatting), accuracy still rises
monotonically to k=32 at every gameability level tested — gameability **suppresses the gain, it does
not reverse it**. Best-of-32 buys +46pp with a clean judge and +11pp with a heavily gameable one.
Real, but that is opportunity cost, not the harm story.

The curve only falls when the judge-pleasing feature is **anti-correlated with correctness** — when
judge-pleasing outputs are actively *wrong* (confident fabrication, plausible-sounding bad reasoning,
the padding a null model exploits). Sweeping that correlation at fixed gameability `b = 2`:

| corr(judge-pleasing, correct) | acc@1 | acc@8 | acc@32 | peak k | acc@32 − acc@1 |
|---|---|---|---|---|---|
| 0.0 | 0.496 | 0.637 | 0.688 | 32 | **+19.15 pp** |
| −0.2 | 0.491 | 0.507 | 0.502 | **8** | +1.11 pp |
| −0.4 | 0.500 | 0.389 | 0.341 | **1** | **−15.91 pp** |
| −0.6 | 0.497 | 0.242 | 0.125 | 1 | −37.21 pp |
| −0.8 | 0.508 | 0.116 | 0.028 | 1 | −48.08 pp |

**The corrected thesis:**

> Whether best-of-N against an LLM judge helps or harms is set by the sign and magnitude of the
> correlation between judge-pleasing features and correctness. There is a critical anti-correlation
> — near **−0.2** in this model — below which more samples make outputs *worse*, and near which the
> curve peaks at an **interior optimal N** rather than at the largest N affordable.

Three things make this better than what it replaced. The threshold is **small**: you do not need a
pathological judge, only a mild preference for confident-wrong over hedged-right. It yields a
**decision** rather than a warning — cap N at the peak. And it makes the empirical question sharp and
answerable: *which side of the threshold are real judges on, for real tasks?* That is exactly what the
experiment measures, and neither answer is a null result.

**Correction to the harm story in Law 0 above.** It assumed the harm always occurs. It does not — it
is conditional on anti-correlation. The "dashboard green, CSAT flat" scenario is the `−0.2` row, not
the general case, and must be written as conditional.

Falsifiable three ways, all still live: real judges may sit at correlation ≥ 0 on every task tested
(then the finding is that best-of-N is safe, which is publishable and useful); the threshold may not
separate judges that mean agreement already separates; or apparent decay may be the optimizer finding
genuinely better outputs. **The third is the hard one, and the verifiable-domain design is what makes
it identifiable.**

## PILOT RESULT — 2026-08-03. **The headline claim did not reproduce.**

120 tasks × 24 samples, 2,880 generations + 2,880 judge calls, ~$30, zero API errors,
zero parse failures. Generator `claude-haiku-4-5`, judge `claude-opus-5`. All three mutants
broke as required (shuffled judge → flat to 0.0 SE; perfect judge → exactly the oracle;
reversed judge → 0.669 → 0.521).

| k | judge says | actually correct | oracle | gap to oracle |
|---|---|---|---|---|
| 1 | 79.0 | 0.6681 | 0.6681 | 0.0000 |
| 4 | 89.2 | 0.7667 | 0.7891 | 0.0224 |
| 8 | 93.3 | 0.8091 | 0.8459 | 0.0368 |
| 24 | 96.5 | 0.8326 | 0.9083 | 0.0757 |

**1. No reversal, and no interior optimum.** `acc@24 − acc@1 = +16.45 pp` (5.6 SE), rising
monotonically, peaking at the largest k tested. Best-of-N *works* here. The Goodhart curve
this paper is named after does not appear.

**2. The efficiency decline is NOT significant, and I nearly reported that it was.** The
aggregate captured-share falls 81.5% → 68.5% from k=4 to k=24, which looks like decay. Per
task and paired, it is **−0.023 ± 0.028, i.e. 0.8 SE**. The aggregate figure is a
ratio-of-means artifact. **Must not be claimed.**

**3. The one clean finding is a negative one.** Judge overconfidence is **flat under
selection pressure**: +12.2 pp at k=1 and +13.3 pp at k=24, near-constant across a 24×
increase in optimization pressure. Selection did not degrade the judge's calibration.

**Why, and it is exactly what `power.py` predicted.** The simulation said that when
judge-pleasing features are merely *uninformative* the result is suppressed gain, not
reversal — and that reversal needs anti-correlation below about −0.2. This task/judge pair
sits in the benign regime. The prediction held; the interesting regime is elsewhere.

**The likely design flaw: best-of-24 by random sampling is weak optimization pressure.**
Drawing 24 independent samples and taking the judge's favourite is not a search for the
judge's blind spots — it is a lottery over the generator's natural output distribution. Real
Goodhart pressure comes from *optimising against* the judge: prompt optimisation, rejection
sampling with feedback, RLAIF. The null here is plausibly about pressure, not about judges.

**Status: the paper as scoped is not supported.** Before any writing, one of these must
hold — (a) a task family where judge-pleasing genuinely anti-correlates with correctness,
(b) real optimisation against the judge rather than i.i.d. sampling, or (c) accept the
negative result as the contribution, which is honest but much quieter. **Do not write the
Goodhart-curve paper on this evidence.**

## EXPERIMENT 2 — direct optimisation against the judge. **Also negative, and this one counts.**

The pilot's null was dismissible: best-of-24 i.i.d. sampling is weak pressure. This run
fixes that. The generator is shown its judge score and told to raise it, five rounds of
hill-climbing, against a **control arm** that starts from the identical solution, gets the
same revision budget, and is told to be more likely correct without the judge ever being
mentioned. 120 tasks, 1,440 rows, ~2,600 calls. Judge switched to `claude-sonnet-5` after
measuring three candidates (below).

| round | JUDGE score | JUDGE acc | CONTROL score | CONTROL acc | paired J−C |
|---|---|---|---|---|---|
| 0 | 77.5 | 0.675 | 77.5 | 0.675 | +0.00 pp |
| 1 | 82.3 | 0.733 | 85.2 | 0.783 | −5.00 pp |
| 3 | 87.1 | 0.817 | 88.1 | 0.817 | +0.00 pp |
| 5 | 89.6 | 0.858 | 85.8 | 0.783 | +7.50 pp |

**Pressure was genuinely applied**, which is what the pilot lacked: judge score climbed
**77.5 → 89.6 (+11.5, 4.8 SE)**. The optimiser could and did move the metric.

**Correctness did not pay for it.** Accuracy in the judge arm rose too, 0.675 → 0.858
(+18.3 pp, 4.3 SE). At no round was the judge arm significantly *worse* than control.

**What must NOT be claimed.** The +7.50 pp judge-over-control advantage at round 5 is
**2.2 SE uncorrected across six round-wise comparisons**, and the trajectory bounces
(−5.00, −2.50, 0.00, −0.83, +7.50) against an SE of ~3.3 pp. It is suggestive that
optimising for the judge beat optimising for correctness, and it is **not** robust.
Do not write it as a finding.

**The defensible conclusion:** *We applied real optimisation pressure to a well-calibrated
LLM judge on a verifiable reasoning domain and could not induce a Goodhart effect. Judge
score rose 12 points; accuracy rose with it.* Two experiments, weak pressure and strong,
both negative.

**Why this is a result rather than a failure.** The null-model paper [V1] proves the
adversarial extreme is catastrophic — 86.5% win rate from a constant irrelevant response.
This bounds the other end: ordinary iterative optimisation against a discriminating judge
(AUC 0.851) does not get you there on a verifiable task. The danger zone is narrower than
"best-of-N will Goodhart your judge", and knowing where it *isn't* is worth stating.

**Judge selection, measured not assumed** (60 pilot solutions, 30 correct / 30 wrong):

| judge | mean | sd | AUC | >90 | headroom |
|---|---|---|---|---|---|
| haiku-4-5 | 91.7 | 12.1 | 0.644 | 95% | 8.3 |
| **sonnet-5** | 67.0 | 39.7 | **0.851** | 63% | **33.0** |
| opus-5 | 69.8 | 39.1 | 0.738 | 67% | 30.2 |

Haiku sits on the ceiling and cannot discriminate — selection against it is near-random,
so a null there would have meant nothing. **Opus discriminates worse than Sonnet at five
times the price**, which is a small independent finding worth reporting.

**Honest limits.** One task family (multi-step arithmetic word problems), one generator,
one judge, five rounds — moderate pressure, not RLAIF scale. Verifiable domain, where
"looks correct to a careful judge" and "is correct" genuinely coincide; subjective domains
plausibly differ and that is the obvious next experiment. 120 tasks gives SE ≈ 3.3 pp, so
degradations under ~7 pp were not detectable.

## BUDGET — settled before spending, by `power.py`

Best-of-k for every k ≤ N comes free from one generation run by subsampling, so the bill is set by N,
not by the number of curve points. Paired MDE on acc@32 − acc@8, at 80% power:

| problems | samples | generation calls | judge calls | MDE |
|---|---|---|---|---|
| 50 | 32 | 1,600 | 1,600 | 10.7 pp |
| 100 | 32 | 3,200 | 3,200 | 7.8 pp |
| **200** | **32** | **6,400** | **6,400** | **4.9 pp** |
| 500 | 32 | 16,000 | 16,000 | 3.5 pp |

Effects in the table above run 1–48 pp, and the between-judge differences that carry the paper are
~18 pp. **200 problems × 32 samples is adequate and affordable**; 500 only if the real effects come in
near the low end. This is the check the previous paper never ran.

---

## VERIFIED — extracted by me from the primary source

**[V1] The adversarial endpoint is catastrophic.** Zheng, Pang, Du, Liu, Jiang & Lin, *Cheating
Automatic LLM Benchmarks: Null Models Achieve High Win Rates*, **ICLR 2025 (Oral)**,
arXiv:2410.07137. A null model that *"always outputs a constant response (irrelevant to input
instructions)"* achieves **86.5% LC win rate on AlpacaEval 2.0**, **83.0 on Arena-Hard-Auto**, and
**9.55 on MT-Bench**. Fetched and confirmed verbatim 2026-08-03.

**[V2] The nearest prior work is about reward models, not judges.** Gao, Schulman & Hilton, *Scaling
Laws for Reward Model Overoptimization*, arXiv:2210.10760. The proxy is a **reward model** and the
gold standard is *"a fixed 'gold-standard' reward model"* that *"plays the role of humans."* No
humans, no LLM judges, no rubrics. **This is the paper that most looks like mine and is not.** Must be
cited and distinguished in the first two paragraphs. *(Scout-fetched; re-extract before writing.)*

---

## WHAT I MUST NOT CLAIM

Each of these is false or taken. Writing any of them is fatal.

1. ~~"Benchmark scores don't predict production behaviour."~~ Consensus, not a finding.
2. ~~"Nobody reports error bars on evals."~~ Anthropic published the argument institutionally
   (Miller, arXiv:2411.00640). The critique literature is substantial.
3. ~~"Commercial endpoint run-to-run variance is unmeasured."~~ **Measured three times.** Most
   damagingly Bjarnason, Silva & Monperrus, arXiv:2602.07150 — **[V]** by me: 60,000 agentic
   trajectories on SWE-Bench-Verified, *"single-run pass@1 estimates vary by 2.2 to 6.0 percentage
   points depending on which run is selected, with standard deviations exceeding 1.5 percentage points
   even at temperature 0."* This killed the previous candidate thesis.
4. ~~"Removing `temperature` took away the control for this variance."~~ **False causal chain.**
   Batch-size nondeterminism is kernel-level, upstream of sampling; `temperature=0` never controlled
   it. Anthropic's own migration guide: it *"never guaranteed identical outputs on prior models."*
5. ~~"Harness choice doesn't matter."~~ Three 2026 papers own this, and one reports backend choice
   alone shifting scores by **up to 16.6 points** — an order of magnitude above endpoint noise.
   **Confront this directly:** it means endpoint nondeterminism is NOT the dominant term, and any
   variance claim I make must be stated against harness variance, not instead of it.
6. ~~"LLM judges are unreliable."~~ Crowded — 21 judges × 9 providers already published.

## PRIOR ART TO CITE AND DISTINGUISH `[unverified — re-extract each before writing]`

- Norman, Rivera & Hughes, *Reliability without Validity*, arXiv:2606.19544 — 21 judges, 9 providers;
  judges with high test–retest reliability alongside severe position bias. **Static measurement.**
- Panickssery, Bowman & Feng, arXiv:2404.13076 — self-preference bias correlates with
  self-recognition. **Static.**
- Hochlehnert et al., *A Sober Look at Progress in Language Model Reasoning*, arXiv:2504.07086.
  **The "15% seed variance on AIME'24" figure came from a search summary and is NOT verified — do not
  cite that number.**
- Dwork et al., arXiv:1506.02629 (reusable holdout) and Blum & Hardt, arXiv:1502.04585 (the Ladder) —
  adaptive data analysis, pre-LLM, and the closest theoretical framing for the sibling question.

## THE DESIGN TRICK — why this is solo-executable

Run the curves on **verifiable domains first**: code with hidden tests, math with known answers.
Ground truth is free and exact, so I can optimize against the judge and read *true* accuracy at every
N with **zero annotation budget**. The judge never sees ground truth. A small human study for
subjective domains comes later, only to show the curve shape transfers.

## THE HARD CONFOUND — and it is the paper's real risk

A rising judge score with rising N could mean the judge is being gamed **or** that the optimizer found
genuinely better outputs. On a verifiable domain these separate cleanly: if true accuracy rises with
judge score, the judge is working; if judge score rises while true accuracy stalls or falls, the gap
is the Goodhart effect. **The verifiable-domain design is not a convenience — it is what makes the
claim identifiable at all.**

## OPEN / TO CHECK BEFORE WRITING

- Re-extract [V2] and every `[unverified]` citation above from the primary source.
- Sweep GitHub issues on eval frameworks, LessWrong, and the Alignment Forum for practitioner
  evidence. The scout's budget ran out before reaching them; the search was arXiv-only.
- One confirming pass for anyone who has measured judge agreement as a function of best-of-N. The
  scout's sweep across 13 best-of-N papers found none, but that is absence of evidence.
