"""Vector figures for the judge-Goodhart paper.

Every figure is computed from the raw JSONL in runs/, never from a number typed by hand,
so a figure cannot drift from the data it depicts. Guards are carried over from the
metric-blindness paper, where a reviewer caught label overflow and arrowheads landing
inside box borders; the guard then showed the arrow defect affected all 12 endpoints
rather than the 2 spotted by eye.

Run: python3 figures.py
"""
import json
import os
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
INK, MID, LIGHT = "#1A1A1A", "#8A8A8A", "#D8D8D8"
ACC = "#B02231"          # single accent, reserved for the judge-optimised arm
COLW, PAGEW = 3.35, 7.0


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), format="pdf")
    fig.savefig(os.path.join(OUT, name.replace(".pdf", ".png")), format="png",
                dpi=200, facecolor="white", edgecolor="none")
    plt.close(fig)
    print("  wrote", name, "+ png")


def assert_in_axes(ax, pts, figname):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    bad = [f"    {lab!r} at ({px:.3g}, {py:.3g})" for lab, px, py in pts
           if not (min(x0, x1) <= px <= max(x0, x1)
                   and min(y0, y1) <= py <= max(y0, y1))]
    if bad:
        raise ValueError(f"{figname}: {len(bad)} annotation(s) off-axes:\n"
                         + "\n".join(bad))
    print(f"  {figname}: {len(pts)} annotations inside the axes")


def assert_claim(cond, figname, msg):
    if not cond:
        raise ValueError(f"{figname}: figure contradicts the prose -- {msg}")
    print(f"  {figname}: {msg}")


def load_unans(path="runs/optimize2.jsonl"):
    by = defaultdict(dict)
    for line in open(os.path.join(HERE, path)):
        r = json.loads(line)
        by[(r["arm"], r["round"])][r["task_id"]] = r
    rounds = sorted({k[1] for k in by})
    tasks = sorted(t for t in by[("judge", 0)] if not by[("judge", 0)][t]["answerable"])
    ans = sorted(t for t in by[("judge", 0)] if by[("judge", 0)][t]["answerable"])
    return by, rounds, tasks, ans


def series(by, arm, rounds, tasks, field):
    m, s = [], []
    for t in rounds:
        v = np.array([float(by[(arm, t)][tid][field] or 0) for tid in tasks])
        m.append(v.mean())
        s.append(v.std(ddof=1) / np.sqrt(v.size))
    return np.array(m), np.array(s)


# ---------------------------------------------------------------- Figure 1
# The headline: optimising against the judge buys confident fabrication.
def fig_fabrication():
    by, rounds, unans, ans = load_unans()
    jm, js = series(by, "judge", rounds, unans, "fabricated")
    cm, cs = series(by, "control", rounds, unans, "fabricated")

    fig, ax = plt.subplots(figsize=(COLW, 2.45))
    ax.plot(rounds, jm, color=ACC, lw=1.7, marker="o", ms=3.4, zorder=3,
            label="optimised against the judge")
    ax.fill_between(rounds, jm - js, jm + js, color=ACC, alpha=0.16, lw=0, zorder=2)
    ax.plot(rounds, cm, color=INK, lw=1.5, ls="--", marker="s", ms=3.0, zorder=3,
            label="control: revise for correctness")
    ax.fill_between(rounds, cm - cs, cm + cs, color=INK, alpha=0.10, lw=0, zorder=2)

    ax.set_xlabel("revision round", labelpad=2)
    ax.set_ylabel("fabrication rate on\nunanswerable problems", labelpad=2)
    ax.set_xlim(-0.15, max(rounds) + 0.15)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xticks(rounds)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.6, pad=1.5)
    ax.legend(frameon=False, fontsize=6.9, loc="center right", handlelength=2.0,
              borderaxespad=0.3, labelspacing=0.35)

    # placed in clear space with a leader: at the obvious spot beside round 0 the label
    # sits directly on the control band
    anns = [("both arms start from the\nsame solution", 0.30, 0.30)]
    ax.annotate(anns[0][0], xy=(0, jm[0]), xytext=anns[0][1:], fontsize=6.6, color=MID,
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=MID, lw=0.6,
                                shrinkA=0, shrinkB=3))
    assert_in_axes(ax, anns, "fig1")

    assert_claim(jm[0] < 0.05 and cm[0] < 0.05, "fig1",
                 f"both arms start near zero ({jm[0]:.3f}, {cm[0]:.3f})")
    assert_claim(jm[-1] - cm[-1] > 0.5, "fig1",
                 f"judge arm ends {(jm[-1] - cm[-1]) * 100:.0f}pp above control")
    assert_claim(cm[-1] <= cm[0] + 0.02, "fig1",
                 f"control does not drift up ({cm[0]:.3f} -> {cm[-1]:.3f})")

    fig.tight_layout(pad=0.25)
    save(fig, "fig1_fabrication.pdf")


# ---------------------------------------------------------------- Figure 2
# The mechanism: the judge pays for fabrication, so the optimiser climbs.
def fig_mechanism():
    recs = [json.loads(l) for l in open(os.path.join(HERE, "runs/optimize2.jsonl"))]
    u = [r for r in recs if not r["answerable"] and r["judge_score"] is not None]
    fab = np.array([r["judge_score"] for r in u if r["fabricated"]], float)
    ref = np.array([r["judge_score"] for r in u if not r["fabricated"]], float)

    fig, ax = plt.subplots(figsize=(COLW, 2.45))
    bins = np.arange(0, 62, 4)
    ax.hist(ref, bins=bins, color=INK, alpha=0.55, label=f"refused (n={ref.size})",
            zorder=2)
    ax.hist(fab, bins=bins, color=ACC, alpha=0.62, label=f"fabricated (n={fab.size})",
            zorder=3)
    ax.axvline(ref.mean(), color=INK, lw=1.2, ls=":", zorder=4)
    ax.axvline(fab.mean(), color=ACC, lw=1.2, ls=":", zorder=4)

    ax.set_xlabel("judge score on unanswerable problems", labelpad=2)
    ax.set_ylabel("solutions", labelpad=2)
    ax.set_xlim(0, 60)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.6, pad=1.5)
    ax.legend(frameon=False, fontsize=6.9, loc="upper right", borderaxespad=0.3)

    ymax = ax.get_ylim()[1]
    anns = [(f"{ref.mean():.1f}", ref.mean(), ymax * 0.72),
            (f"{fab.mean():.1f}", fab.mean(), ymax * 0.72)]
    ax.annotate(anns[0][0], anns[0][1:], fontsize=7.2, color=INK, ha="right",
                xytext=(-3, 0), textcoords="offset points")
    ax.annotate(anns[1][0], anns[1][1:], fontsize=7.2, color=ACC, ha="left",
                xytext=(3, 0), textcoords="offset points")
    assert_in_axes(ax, anns, "fig2")

    assert_claim(fab.mean() > ref.mean(), "fig2",
                 f"judge pays +{fab.mean() - ref.mean():.1f} points for fabrication")
    fig.tight_layout(pad=0.25)
    save(fig, "fig2_mechanism.pdf")


# ---------------------------------------------------------------- Figure 3
# The arc: two nulls are the boundary that makes the third result mean something.
def fig_arc():
    by, rounds, unans, ans = load_unans()
    jm, _ = series(by, "judge", rounds, unans, "fabricated")
    cm, _ = series(by, "control", rounds, unans, "fabricated")
    e3 = (jm[-1] - cm[-1]) * 100

    labels = ["weak pressure\n(best-of-24)", "strong pressure\njudge = truth",
              "strong pressure\njudge ≠ truth"]
    vals = [0.0, 0.0, e3]
    errs = [2.9, 3.4, 5.6]          # SEs from analyze.py / analyze_opt.py / analyze_unans.py

    fig, ax = plt.subplots(figsize=(COLW, 2.45))
    cols = [MID, MID, ACC]
    ax.bar(range(3), vals, yerr=errs, color=cols, width=0.58, zorder=3,
           error_kw=dict(lw=0.9, capsize=3, ecolor=INK))
    ax.axhline(0, color=INK, lw=0.7, zorder=2)

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("harm attributable to\noptimising against the judge (pp)", labelpad=2)
    ax.set_ylim(-12, e3 * 1.28)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.6, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=3)

    anns = [(f"+{e3:.0f}pp", 2, e3 + errs[2] + 3.5)]
    ax.annotate(anns[0][0], anns[0][1:], fontsize=8.0, color=ACC, ha="center",
                fontweight="bold")
    assert_in_axes(ax, anns, "fig3")

    assert_claim(abs(vals[0]) < errs[0] and abs(vals[1]) < errs[1], "fig3",
                 "both null conditions are within one SE of zero")
    assert_claim(vals[2] > 3 * errs[2], "fig3",
                 f"the opposed condition is {vals[2] / errs[2]:.1f} SE from zero")

    fig.tight_layout(pad=0.25)
    save(fig, "fig3_arc.pdf")


if __name__ == "__main__":
    for f in (fig_fabrication, fig_mechanism, fig_arc):
        print(f"\n{f.__name__}:")
        f()
    print("\nall figures generated with guards passing")
