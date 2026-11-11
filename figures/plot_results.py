"""Plot mAP (Medium/Hard, single-scale, +Lw) for frozen DINOv2 and B0..B4. Usage: python figures/plot_results.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, ROxf M, ROxf H, RPar M, RPar H) -- from results/*/results*.csv and results/eval_v2/frozen
ROWS = [
    ("Frozen", 70.1, 49.8, 88.6, 76.0),
    ("Frozen+aQE", 72.2, 51.9, 89.1, 76.6),
    ("B0", 65.2, 45.3, 87.2, 73.3),
    ("B1", 72.1, 52.9, 89.7, 76.5),
    ("B2", 71.7, 52.8, 89.7, 76.0),
    ("B3", 72.0, 52.8, 89.8, 77.1),
    ("B4 (B3+aQE)", 74.1, 54.6, 90.2, 77.7),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
labels = [r[0] for r in ROWS]
x = range(len(ROWS))
for ax, (name, i) in zip(axes, [("ROxf", 1), ("RPar", 3)]):
    m = [r[i] for r in ROWS]; h = [r[i + 1] for r in ROWS]
    ax.bar([k - 0.2 for k in x], m, 0.4, label="Medium", color="#2a6fdb")
    ax.bar([k + 0.2 for k in x], h, 0.4, label="Hard", color="#e07a1f")
    for k, (a, b) in enumerate(zip(m, h)):
        ax.text(k - 0.2, a + 0.5, f"{a:.1f}", ha="center", fontsize=7)
        ax.text(k + 0.2, b + 0.5, f"{b:.1f}", ha="center", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_title(f"{name} mAP (+Lw)"); ax.set_ylim(40 if name == "ROxf" else 70, 95)
    ax.grid(axis="y", alpha=.3)
axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout()
out = Path(__file__).with_name("results.png")
fig.savefig(out, dpi=150)
print(out)
