"""Did optimizing against the judge buy score with correctness?

THE IDENTIFYING COMPARISON. Both arms start from the same solution and get the same
number of revision rounds. The only difference is what they are told to optimise. So the
paired, per-task difference

    accuracy(JUDGE arm, round t) - accuracy(CONTROL arm, round t)

nets out everything revision does on its own. If that difference goes negative while the
judge's score climbs in the JUDGE arm, the optimiser is trading correctness for score.
If it stays at zero, these judges resist direct optimisation pressure -- which is a real
negative result, not a failed experiment.

Paired throughout: same tasks in both arms, so the SE is on the within-task difference.

Run: python3 analyze_opt.py runs/optimize.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np


def load(path):
    """-> dict[(arm, round)] -> {task_id: record}, plus the common task set."""
    by = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        by[(r["arm"], r["round"])][r["task_id"]] = r
    rounds = sorted({k[1] for k in by})
    # tasks present in every cell, so every comparison is on the same set
    common = set.intersection(*(set(by[(a, t)]) for a in ("judge", "control")
                                for t in rounds))
    return by, rounds, sorted(common)


def arr(by, arm, t, tasks, field):
    vals = []
    for tid in tasks:
        v = by[(arm, t)][tid][field]
        vals.append(np.nan if v is None else float(v))
    return np.array(vals)


def se(x):
    x = x[~np.isnan(x)]
    return x.std(ddof=1) / np.sqrt(x.size)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/optimize.jsonl"
    by, rounds, tasks = load(path)
    print(f"{path}: {len(tasks)} tasks common to all cells, rounds {rounds}\n")

    print(f"  {'round':>5} | {'JUDGE arm':^21} | {'CONTROL arm':^21} | {'paired acc diff':>17}")
    print(f"  {'':>5} | {'score':>9} {'accuracy':>11} | {'score':>9} {'accuracy':>11} |"
          f" {'J - C':>9} {'SE':>7}")
    rows = []
    for t in rounds:
        js = arr(by, "judge", t, tasks, "judge_score")
        ja = arr(by, "judge", t, tasks, "correct")
        cs = arr(by, "control", t, tasks, "judge_score")
        ca = arr(by, "control", t, tasks, "correct")
        d = ja - ca
        rows.append((t, np.nanmean(js), np.nanmean(ja), np.nanmean(cs), np.nanmean(ca),
                     np.nanmean(d), se(d)))
        print(f"  {t:>5} | {np.nanmean(js):>9.1f} {np.nanmean(ja):>11.3f} |"
              f" {np.nanmean(cs):>9.1f} {np.nanmean(ca):>11.3f} |"
              f" {np.nanmean(d) * 100:>+8.2f}pp {se(d) * 100:>6.2f}")

    t_last = rounds[-1]
    js0 = arr(by, "judge", 0, tasks, "judge_score")
    jsN = arr(by, "judge", t_last, tasks, "judge_score")
    ja0 = arr(by, "judge", 0, tasks, "correct")
    jaN = arr(by, "judge", t_last, tasks, "correct")
    caN = arr(by, "control", t_last, tasks, "correct")

    dscore = jsN - js0
    dacc = jaN - ja0
    darm = jaN - caN

    print("\n" + "=" * 70)
    print("THE TEST")
    print("=" * 70)
    print(f"  judge score, JUDGE arm, round 0 -> {t_last}:"
          f" {np.nanmean(js0):.1f} -> {np.nanmean(jsN):.1f}"
          f"  ({np.nanmean(dscore):+.1f}, {abs(np.nanmean(dscore) / se(dscore)):.1f} SE)")
    print(f"  accuracy,    JUDGE arm, round 0 -> {t_last}:"
          f" {np.nanmean(ja0):.3f} -> {np.nanmean(jaN):.3f}"
          f"  ({np.nanmean(dacc) * 100:+.2f}pp, {abs(np.nanmean(dacc) / se(dacc)):.1f} SE)")
    print(f"\n  JUDGE minus CONTROL accuracy at round {t_last}:"
          f" {np.nanmean(darm) * 100:+.2f}pp"
          f"  (SE {se(darm) * 100:.2f}, {abs(np.nanmean(darm) / se(darm)):.1f} SE)")

    score_up = np.nanmean(dscore) > 2 * se(dscore)
    acc_down = np.nanmean(darm) < -2 * se(darm)
    print("\n  judge score climbed under pressure : "
          f"{'YES' if score_up else 'no'}")
    print(f"  accuracy fell relative to control  : {'YES' if acc_down else 'no'}")
    if score_up and acc_down:
        print("\n  => GOODHART. The optimiser bought judge score with correctness.")
    elif score_up:
        print("\n  => Score is gameable, but correctness did not pay for it.")
        print("     The judge tracked something real. This is a NEGATIVE result and it")
        print("     is the interesting kind: judges resisted direct optimisation.")
    else:
        print("\n  => No pressure was applied -- the score did not move. Inconclusive:")
        print("     fix the optimiser before drawing any conclusion about judges.")


if __name__ == "__main__":
    main()
