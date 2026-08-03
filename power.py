"""Is the Goodhart curve detectable at a budget I can actually afford?

WHY THIS RUNS FIRST. The previous paper (deferral-coupling) was killed after two days
partly because its central effect could not be measured at any sample size the field
runs. That check cost nothing and was never done. This file is that check, done first,
before a single API call is paid for.

THE QUESTION. With P problems x N samples per problem, and a judge that is partly valid
and partly gameable, can I distinguish a real decay in best-of-k accuracy from sampling
noise -- and how big must the decay be?

THE MODEL. For problem i with per-sample success probability p_i:
    correct_ij  ~ Bernoulli(p_i)                      ground truth, free on a verifiable domain
    spurious_ij ~ Normal(0, 1)                        a judge-pleasing feature uncorrelated with truth
    judge_ij    = a*correct_ij + b*spurious_ij + eps  the judge sees neither p_i nor truth

    best-of-k   = argmax_j judge_ij over a random k-subset
    accuracy(k) = P(the selected sample is actually correct)

`a` is judge VALIDITY (how much its score tracks truth); `b` is judge GAMEABILITY (how
much it tracks something a selector can exploit). The Goodhart claim is that b > 0 makes
accuracy(k) stall or fall while judge score keeps climbing -- and that the gap widens
with k, because best-of-k is a search for high-b samples.

WHY BEST-OF-K IS FREE. One generation budget buys the whole curve: draw N samples once,
then estimate best-of-k for every k <= N by subsampling. The API cost is set by N, not
by the number of curve points.

MUTATION DISCIPLINE (Law 2). Every claim has a mutant that must break it:
  - b = 0 (valid, ungameable judge) MUST give accuracy rising monotonically in k.
  - a = 0 (judge carries no signal about truth) MUST give accuracy flat at the base rate.
  - judge scores shuffled across samples MUST give accuracy flat at the base rate.
If any mutant passes, the simulation is measuring argmax, not Goodhart.

Run: python3 power.py
"""
import numpy as np

SEED = 17
N_SAMPLES = 32           # generations per problem -- this sets the API bill
KS = (1, 2, 4, 8, 16, 32)


def simulate(rng, n_problems, n_samples=N_SAMPLES, a=1.0, b=0.0, noise=0.5,
             diff_a=2.0, diff_b=2.0, shuffle_judge=False, spur_corr=0.0):
    """Return (correct, judge) arrays of shape (n_problems, n_samples).

    `spur_corr` is the correlation between the judge-pleasing feature and actual
    correctness. Zero means the feature is merely uninformative (verbosity, formatting).
    NEGATIVE means judge-pleasing outputs are actively WORSE -- confident fabrication,
    plausible-sounding wrong reasoning, the padding a null model exploits. The ICLR
    null-model result (86.5% win rate from a constant irrelevant response) is the
    extreme of this regime, so a model that cannot express it cannot describe reality.
    """
    p = rng.beta(diff_a, diff_b, n_problems)[:, None]        # per-problem success rate
    correct = (rng.random((n_problems, n_samples)) < p).astype(float)
    z = rng.standard_normal((n_problems, n_samples))
    if spur_corr == 0.0:
        spurious = z
    else:
        # standardise correctness within problem, then mix to hit the target correlation
        cs = correct - correct.mean(axis=1, keepdims=True)
        sd = cs.std(axis=1, keepdims=True)
        cs = np.divide(cs, sd, out=np.zeros_like(cs), where=sd > 1e-9)
        spurious = spur_corr * cs + np.sqrt(max(0.0, 1 - spur_corr ** 2)) * z
    judge = a * correct + b * spurious + noise * rng.standard_normal((n_problems, n_samples))
    if shuffle_judge:
        # break the judge<->sample correspondence, keeping the marginal distribution
        flat = judge.ravel().copy()
        rng.shuffle(flat)
        judge = flat.reshape(judge.shape)
    return correct, judge


def accuracy_at_k(correct, judge, k, rng, reps=64):
    """Accuracy of the judge's best-of-k pick, per problem, averaged over random
    k-subsets. Returns a per-problem array so the SE can be taken across problems."""
    n_problems, n_samples = correct.shape
    if k == 1:
        return correct.mean(axis=1)                      # exact: a random single draw
    hits = np.zeros(n_problems)
    for _ in range(reps):
        idx = np.argsort(rng.random((n_problems, n_samples)), axis=1)[:, :k]
        j = np.take_along_axis(judge, idx, axis=1)
        c = np.take_along_axis(correct, idx, axis=1)
        pick = np.argmax(j, axis=1)
        hits += c[np.arange(n_problems), pick]
    return hits / reps


def curve(correct, judge, rng, ks=KS):
    """Per-problem accuracy at each k. Shape (len(ks), n_problems)."""
    return np.array([accuracy_at_k(correct, judge, k, rng) for k in ks])


def se_of_difference(per_problem_a, per_problem_b):
    """SE of the paired difference in accuracy between two k values, across problems.
    Paired because the same problems appear at both k -- this is what a real analysis
    would do, and it is meaningfully tighter than treating them as independent."""
    d = per_problem_a - per_problem_b
    return d.std(ddof=1) / np.sqrt(d.size)


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 74)
    print("1. DOES THE MODEL BEHAVE? (mutants must fail)")
    print("=" * 74)

    # -- mutant A: ungameable judge -> accuracy must RISE with k
    c, j = simulate(rng, 400, a=1.0, b=0.0)
    cv = curve(c, j, rng).mean(axis=1)
    print(f"  b=0  (valid, ungameable)   {np.array2string(cv, precision=3)}")
    rises = np.all(np.diff(cv) > -1e-3)
    print(f"       monotone rising: {rises}  <- must be True")
    assert rises, "an ungameable judge must improve accuracy with more samples"

    # -- mutant B: judge carries no truth signal -> flat at base rate
    #
    # "Flat" must be judged against the standard error, not an arbitrary constant.
    # The first version of this check used |max-min| < 0.02 and reported a FALSE
    # FAILURE on mutant C: a 2pp swing at n=400, where the SE is ~2.5pp, is flat.
    # A threshold that does not know its own noise floor is not a test -- which is,
    # with some irony, this paper's entire thesis.
    def assert_flat(per_problem_curve, label, n_se=3.0):
        se = se_of_difference(per_problem_curve[-1], per_problem_curve[0])
        diff = per_problem_curve[-1].mean() - per_problem_curve[0].mean()
        ok = abs(diff) < n_se * se
        print(f"       k=1 -> k=32 change {diff * 100:+.2f} pp, SE {se * 100:.2f} pp"
              f"  ({abs(diff) / se:.1f} SE)  flat: {ok}  <- must be True")
        assert ok, f"{label}: expected no selection signal, got {diff / se:.1f} SE"

    c, j = simulate(rng, 400, a=0.0, b=1.0)
    cv0_pp = curve(c, j, rng)
    print(f"  a=0  (no truth signal)     {np.array2string(cv0_pp.mean(axis=1), precision=3)}"
          f"   base rate {c.mean():.3f}")
    assert_flat(cv0_pp, "judge with no truth signal")

    # -- mutant C: shuffled judge -> flat
    c, j = simulate(rng, 400, a=1.0, b=0.0, shuffle_judge=True)
    cvs_pp = curve(c, j, rng)
    print(f"  shuffled judge scores      {np.array2string(cvs_pp.mean(axis=1), precision=3)}")
    assert_flat(cvs_pp, "shuffled judge scores")

    print("\n" + "=" * 74)
    print("2. THE EFFECT: how gameability bends the curve")
    print("=" * 74)
    print(f"  {'gameability b':>14}  {'acc@1':>7} {'acc@8':>7} {'acc@32':>7}  "
          f"{'peak k':>7}  {'shape':>10}")
    for b in (0.0, 0.5, 1.0, 2.0, 4.0):
        c, j = simulate(rng, 800, a=1.0, b=b)
        cv = curve(c, j, rng).mean(axis=1)
        peak = KS[int(np.argmax(cv))]
        shape = "rising" if peak == KS[-1] else ("falls" if cv[-1] < cv.max() - 0.01 else "plateau")
        print(f"  {b:>14.1f}  {cv[0]:>7.3f} {cv[3]:>7.3f} {cv[-1]:>7.3f}  "
              f"{peak:>7}  {shape:>10}")

    print("\n" + "=" * 74)
    print("2b. CAN THE CURVE ACTUALLY FALL? (the decisive design question)")
    print("=" * 74)
    print("  Above, gameability only SUPPRESSES the gain -- accuracy still rises to")
    print("  k=32 at every b. A genuine Goodhart curve requires the judge-pleasing")
    print("  feature to be ANTI-CORRELATED with correctness. Sweeping that:\n")
    print(f"  {'corr(spur,correct)':>19}  {'acc@1':>7} {'acc@8':>7} {'acc@32':>7}"
          f"  {'peak k':>7}  {'@32 vs @1':>10}")
    falls_somewhere = False
    for sc in (0.0, -0.2, -0.4, -0.6, -0.8):
        c, j = simulate(rng, 800, a=1.0, b=2.0, spur_corr=sc)
        cv = curve(c, j, rng).mean(axis=1)
        peak = KS[int(np.argmax(cv))]
        delta = (cv[-1] - cv[0]) * 100
        if cv[-1] < cv[0] - 0.01:
            falls_somewhere = True
        print(f"  {sc:>19.1f}  {cv[0]:>7.3f} {cv[3]:>7.3f} {cv[-1]:>7.3f}"
              f"  {peak:>7}  {delta:>+9.2f}pp")
    print(f"\n  Curve drops below its k=1 value somewhere: {falls_somewhere}")
    print("  If False, the honest headline is SUPPRESSED GAIN, not decay, and the")
    print("  claim sheet's 'dashboard green while outcomes flat' story must be cut.")

    print("\n" + "=" * 74)
    print("3. POWER: how many problems to detect the bend?")
    print("=" * 74)
    print("  Testing acc@32 vs acc@8 (paired). MDE = smallest true gap detectable")
    print("  at 80% power, two-sided alpha=0.05  ->  approx 2.8 x SE.\n")
    print(f"  {'problems':>9} {'samples':>8} {'gen calls':>10} {'judge calls':>12}"
          f" {'SE':>8} {'MDE (pp)':>9}")
    for n_problems in (50, 100, 200, 500):
        c, j = simulate(rng, n_problems, a=1.0, b=1.0)
        cv = curve(c, j, rng)
        se = se_of_difference(cv[-1], cv[3])
        calls = n_problems * N_SAMPLES
        print(f"  {n_problems:>9} {N_SAMPLES:>8} {calls:>10,} {calls:>12,}"
              f" {se:>8.4f} {2.8 * se * 100:>9.2f}")

    print("\n  For reference, the true acc@32 - acc@8 gap at each gameability level:")
    for b in (0.5, 1.0, 2.0, 4.0):
        c, j = simulate(rng, 2000, a=1.0, b=b)
        cv = curve(c, j, rng).mean(axis=1)
        print(f"    b={b:.1f}  true gap = {(cv[-1] - cv[3]) * 100:+.2f} pp")

    print("\n" + "=" * 74)
    print("4. THE HARDER TEST: do judges DIFFER in decay rate?")
    print("=" * 74)
    print("  The paper's distinctive claim is that mean agreement at k=1 does NOT")
    print("  predict the decay. Two judges matched at k=1, different b:\n")
    # tune `a` so both judges have near-identical accuracy at k=1... which is exact by
    # construction (k=1 is a random draw), so match them on acc@2 instead -- the first
    # point where the judge's ranking actually does any work.
    for label, (a, b) in {"judge A (low gameability) ": (1.0, 0.5),
                          "judge B (high gameability)": (1.6, 4.0)}.items():
        c, j = simulate(rng, 1500, a=a, b=b)
        cv = curve(c, j, rng).mean(axis=1)
        print(f"  {label}  acc@2={cv[1]:.3f}  acc@32={cv[-1]:.3f}"
              f"  delta={(cv[-1] - cv[1]) * 100:+.2f} pp")
    print("\n  If these two can be matched at low k and separated at high k, the")
    print("  'flatness beats mean agreement' claim is measurable. If they cannot,")
    print("  the paper's selection advice has no support and must be dropped.")

    print("\nall assertions passed -- including that the mutants fail")


if __name__ == "__main__":
    main()
