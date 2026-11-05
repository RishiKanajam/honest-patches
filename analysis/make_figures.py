"""Reproduce the Honest Patches results table and figures from lcb_runs.jsonl.

CPU only. Usage: python make_figures.py lcb_runs.jsonl
"""
import json, sys
from collections import Counter
from math import sqrt
import matplotlib.pyplot as plt

sys.path.insert(0, "runners")
from lcb_runner import detect_tactics

LOG = sys.argv[1] if len(sys.argv) > 1 else "logs/lcb_runs.jsonl"
rows = [json.loads(l) for l in open(LOG)]
# Every logged run is used; bars show k/n so differing seed counts stay visible.
extra = []
NO_TASK = {"test": "", "prompt": ""}


def attempted(r):
    return any("call_counter_state" in detect_tactics(a["code"], NO_TASK) for a in r["attempts"])


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0, c - h), c + h


CONDS = [("baseline", "No note"), ("injection_soft", "Note: \u201ctests are flaky\u201d"),
         ("injection", "Note: flaky + \u201cspecial-casing\nis acceptable\u201d"),
         ("injection_guard", "Same note +\nsystem-prompt guard")]

print(f"{'condition':<16}{'runs':>6}{'passed':>9}{'attempted':>11}")
stats = []
for cond, label in CONDS:
    rs = [r for r in rows if r["split"] == "conflicting" and r["condition"] == cond]
    k_pass, k_att, n = sum(r["final_passed"] for r in rs), sum(attempted(r) for r in rs), len(rs)
    stats.append((label, n, wilson(k_pass, n), wilson(k_att, n), k_pass, k_att))
    print(f"{cond:<16}{n:>6}{k_pass:>9}{k_att:>11}")
for cond in sorted({r["condition"] for r in extra}):
    rs = [r for r in extra if r["condition"] == cond]
    print(f"extra seeds, {cond}: {sum(r['final_passed'] for r in rs)} passed, "
          f"{sum(attempted(r) for r in rs)} attempted, of {len(rs)}")
orig = [r for r in rows if r["split"] == "original"]
print(f"original solved honestly: {sum(r['final_passed'] for r in orig)}/{len(orig)}")

# Figure 1: attempted vs successful cheating per condition, with 95% Wilson intervals.
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = range(len(stats))
w = 0.36
for off, idx, name, color in [(-w / 2, 3, "Attempted cheating", "#E8A33D"),
                              (w / 2, 2, "Successful cheating (passed)", "#C0392B")]:
    vals = [s[idx][0] * 100 for s in stats]
    errs = [[(s[idx][0] - s[idx][1]) * 100 for s in stats], [(s[idx][2] - s[idx][0]) * 100 for s in stats]]
    bars = ax.bar([i + off for i in x], vals, w, yerr=errs, capsize=4, color=color, label=name)
    counts = [s[5] if idx == 3 else s[4] for s in stats]
    for b, k, s in zip(bars, counts, stats):
        top = s[idx][2] * 100  # place the label above the error bar
        ax.text(b.get_x() + b.get_width() / 2, top + 0.8, f"{k}/{s[1]}",
                ha="center", va="bottom", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels([s[0] for s in stats])
ax.set_ylabel("Runs (%)")
ax.set_ylim(0, 36)
ax.set_title("Permission, not doubt: a planted authorization makes Gemma 4 31B cheat,\nand a prompt guard only partly stops it",
             fontsize=13, fontweight="bold")
ax.legend(frameon=False, loc="upper left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.text(0.01, 0.005, "Impossible-LiveCodeBench, conflicting split (103 tasks). Error bars: 95% Wilson intervals.",
         fontsize=8.5, color="#555")
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("fig1_cheating_by_condition.png", dpi=200)

# Figure 2: the attempt at which cheating first appears (injection condition).
first = Counter()
for r in rows:
    if r["split"] == "conflicting" and r["condition"] == "injection" and attempted(r):
        first[min(i + 1 for i, a in enumerate(r["attempts"])
                  if "call_counter_state" in detect_tactics(a["code"], NO_TASK))] += 1
fig, ax = plt.subplots(figsize=(6, 3.6))
ks = [1, 2, 3]
bars = ax.bar([f"Attempt {k}" for k in ks], [first[k] for k in ks], color="#E8A33D")
for b, k in zip(bars, ks):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, str(first[k]), ha="center")
ax.set_ylabel("Runs that started cheating")
after, total = sum(first[k] for k in (2, 3)), sum(first.values())
lead = f"All {total}" if after == total else f"{after} of {total}"
ax.set_title(f"{lead} cheating attempts began only after a test failure", fontsize=11, fontweight="bold")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
plt.savefig("fig2_first_cheating_attempt.png", dpi=200)
print("saved fig1_cheating_by_condition.png, fig2_first_cheating_attempt.png")
