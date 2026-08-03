"""Read the raw JSONL and compute the best-of-k curve, with mutants that must fail.

THE THREE CURVES, and the paper lives in the gap between them:

  judge@k   the judge's own score for its best-of-k pick. Rises by construction --
            selecting the max of k draws from any distribution goes up with k. This
            is what a team watches on a dashboard, and it can only ever go up.
  acc@k     whether that pick was ACTUALLY correct. This is what the team wants and
            cannot see.
  oracle@k  best achievable: correct if ANY of the k samples was correct. The ceiling
            a perfect judge would reach.

If acc@k tracks oracle@k, the judge is doing real work. If judge@k climbs while acc@k
stalls below oracle@k, the selector is finding samples the judge likes and the world
does not -- the Goodhart gap. If acc@k falls below acc@1, best-of-N is actively
harmful and there is an interior optimal k.

MUTANTS (Law 2). Each must break, and the script fails loudly if one does not:
  - shuffling judge scores across samples must flatten acc@k to the base rate
  - a perfect judge (score = correctness) must make acc@k equal oracle@k
  - reversing the judge must make acc@k fall
A curve that survives all three is measuring selection, not argsort.

Run: python3 analyze.py runs/pilot.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

KS = (1, 2, 4, 8, 16, 24)
REPS = 256
SEED = 5


def load(path):
    """-> (correct, judge) float arrays of shape (n_tasks, n_samples), plus task ids.
    Rows with a missing judge score or unparsed answer are dropped, and tasks are
    truncated to a common sample count so the array is rectangular."""
    by_task = defaultdict(list)
    dropped = 0
    for line in open(path):
        r = json.loads(line)
        if r["judge_score"] is None or str(r["solution"]).startswith("__ERROR__"):
            dropped += 1
            continue
        by_task[r["task_id"]].append((float(r["correct"]), float(r["judge_score"])))
    n = min(len(v) for v in by_task.values())
    ids = sorted(by_task)
    correct = np.array([[c for c, _ in by_task[t][:n]] for t in ids])
    judge = np.array([[j for _, j in by_task[t][:n]] for t in ids])
    return correct, judge, ids, dropped


def curves(correct, judge, rng, ks=KS, reps=REPS):
    """Per-task acc@k, judge@k and oracle@k. Each shape (len(ks), n_tasks)."""
    n_tasks, n_samples = correct.shape
    acc, jsc, orc = [], [], []
    for k in ks:
        if k > n_samples:
            continue
        a = np.zeros(n_tasks)
        j = np.zeros(n_tasks)
        o = np.zeros(n_tasks)
        for _ in range(reps):
            idx = np.argsort(rng.random((n_tasks, n_samples)), axis=1)[:, :k]
            jk = np.take_along_axis(judge, idx, axis=1)
            ck = np.take_along_axis(correct, idx, axis=1)
            pick = np.argmax(jk, axis=1)
            rows = np.arange(n_tasks)
            a += ck[rows, pick]
            j += jk[rows, pick]
            o += ck.max(axis=1)
        acc.append(a / reps)
        jsc.append(j / reps)
        orc.append(o / reps)
    return np.array(acc), np.array(jsc), np.array(orc)


def se(per_task):
    return per_task.std(ddof=1) / np.sqrt(per_task.shape[0])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/pilot.jsonl"
    rng = np.random.default_rng(SEED)
    correct, judge, ids, dropped = load(path)
    n_tasks, n_samples = correct.shape
    ks = [k for k in KS if k <= n_samples]

    print(f"{path}: {n_tasks} tasks x {n_samples} usable samples "
          f"({dropped} rows dropped)")
    print(f"base accuracy (any single sample): {correct.mean():.4f}\n")

    acc, jsc, orc = curves(correct, judge, rng, ks)

    print(f"  {'k':>4} {'judge@k':>9} {'acc@k':>9} {'+/-':>7} {'oracle@k':>9}"
          f" {'gap to oracle':>14}")
    for i, k in enumerate(ks):
        print(f"  {k:>4} {jsc[i].mean():>9.2f} {acc[i].mean():>9.4f}"
              f" {se(acc[i]):>7.4f} {orc[i].mean():>9.4f}"
              f" {orc[i].mean() - acc[i].mean():>14.4f}")

    d = acc[-1] - acc[0]
    print(f"\n  acc@{ks[-1]} - acc@1 = {d.mean() * 100:+.2f} pp "
          f"(SE {se(d) * 100:.2f}, {abs(d.mean() / se(d)):.1f} SE)")
    peak = ks[int(np.argmax(acc.mean(axis=1)))]
    print(f"  judge score rose {jsc[0].mean():.1f} -> {jsc[-1].mean():.1f}")
    print(f"  accuracy peaks at k = {peak}"
          f"{'  <- INTERIOR OPTIMUM' if peak not in (ks[0], ks[-1]) else ''}")

    print("\n" + "=" * 68)
    print("MUTANTS (each must break the effect)")
    print("=" * 68)

    # 1. shuffled judge -> selection carries no information -> flat at base rate
    flat = judge.ravel().copy()
    rng.shuffle(flat)
    sacc, _, _ = curves(correct, flat.reshape(judge.shape), rng, ks)
    diff = sacc[-1] - sacc[0]
    ok = abs(diff.mean()) < 3 * se(diff)
    print(f"  shuffled judge     acc@1 {sacc[0].mean():.4f} -> acc@{ks[-1]} "
          f"{sacc[-1].mean():.4f}  ({abs(diff.mean() / se(diff)):.1f} SE)  "
          f"flat: {ok}")
    assert ok, "shuffled judge scores must carry no selection signal"

    # 2. perfect judge -> acc must equal oracle
    pacc, _, porc = curves(correct, correct * 100.0, rng, ks)
    match = np.allclose(pacc.mean(axis=1), porc.mean(axis=1), atol=1e-9)
    print(f"  perfect judge      acc@{ks[-1]} {pacc[-1].mean():.4f} vs oracle "
          f"{porc[-1].mean():.4f}  equal: {match}")
    assert match, "a judge that sees correctness must reach the oracle"

    # 3. reversed judge -> accuracy must fall
    racc, _, _ = curves(correct, -judge, rng, ks)
    falls = racc[-1].mean() < racc[0].mean()
    print(f"  reversed judge     acc@1 {racc[0].mean():.4f} -> acc@{ks[-1]} "
          f"{racc[-1].mean():.4f}  falls: {falls}")
    assert falls, "inverting the judge must make selection worse"

    print("\nall mutants failed as required")


if __name__ == "__main__":
    main()
