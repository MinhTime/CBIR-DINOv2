"""Generate result figures for the paper from the real result CSVs / Kaggle logs.
Run from the paper/ folder:  python figures/make_figures.py
All numbers below are copied from cbir/b0_finetune_results.csv, cbir/results/B1..B3."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
C = {"M": "#2a6fb0", "H": "#d0662b", "ref": "#555555"}

# ---- Fig: progression B0 -> B4 (best descriptor, +Lw, single-scale) ----
vers = ["Frozen", "B0\n(GeM)", "B0\n(CLS)", "B1", "B2", "B3", "B4"]
rox = {"M": [70.06, 64.30, 69.28, 72.10, 71.67, 71.99, 74.09], "H": [49.84, 44.17, 48.19, 52.88, 52.77, 52.77, 54.60]}
rpa = {"M": [88.64, 85.87, 88.30, 89.72, 89.70, 89.83, 90.17], "H": [75.96, 71.60, 74.73, 76.46, 76.04, 77.06, 77.73]}
ref = {"rox": (72.17, 51.85), "rpa": (89.06, 76.62)}  # frozen DINOv2 + aQE, our protocol (results/eval_v2/frozen)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, d, name, key in [(axes[0], rox, r"$\mathcal{R}$Oxford5k", "rox"), (axes[1], rpa, r"$\mathcal{R}$Paris6k", "rpa")]:
    x = np.arange(len(vers)); w = 0.38
    cols_M = [C["M"]] * len(vers)
    for k, off in [("M", -w/2), ("H", w/2)]:
        b = ax.bar(x + off, d[k], w, color=C[k], label={"M": "Medium", "H": "Hard"}[k])
        for xi, v in zip(x + off, d[k]): ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5)
    ax.axhline(ref[key][0], color=C["ref"], ls="--", lw=0.9, label="Frozen + $\\alpha$QE, Medium")
    ax.axhline(ref[key][1], color=C["ref"], ls=":", lw=1.1, label="Frozen + $\\alpha$QE, Hard")
    ax.set_xticks(x); ax.set_xticklabels(vers, fontsize=7); ax.set_title(name); ax.set_ylabel("mAP (%)")
    ax.set_ylim(35 if key == "rox" else 65, 80 if key == "rox" else 95)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/progression.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig: training loss curves (from Kaggle logs) ----
ep = np.arange(1, 11)
loss = {
 "B1 (CLS, 2 blocks, lr 2e-6)": [0.1571, 0.1381, 0.1299, 0.1231, 0.1200, 0.1092, 0.1087, 0.1036, 0.0995, 0.0991],
 "B2 (CLS, 4 blocks, lr 2e-6)": [0.1421, 0.1219, 0.1107, 0.1037, 0.0989, 0.0884, 0.0872, 0.0820, 0.0762, 0.0757],
 "B3 (CLS, 2 blocks, lr 1e-6)": [0.1668, 0.1496, 0.1425, 0.1356, 0.1341, 0.1236, 0.1236, 0.1187, 0.1160, 0.1159]}
roxM = {"B1 (CLS, 2 blocks, lr 2e-6)": 72.10, "B2 (CLS, 4 blocks, lr 2e-6)": 71.67, "B3 (CLS, 2 blocks, lr 1e-6)": 71.99}
valM = {"B1 (CLS, 2 blocks, lr 2e-6)": 93.13, "B2 (CLS, 4 blocks, lr 2e-6)": 93.56, "B3 (CLS, 2 blocks, lr 1e-6)": 92.73}  # rSfM-120k val, CLS+Lw
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.5, 1, 1]})
cols = ["#2a6fb0", "#d0662b", "#3a9a5b"]
for (k, v), c, mk in zip(loss.items(), cols, "osD"):
    axes[0].plot(ep, v, marker=mk, ms=3, color=c, label=k)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("contrastive loss"); axes[0].legend(frameon=False, fontsize=6)
axes[0].set_xticks(ep)
fl = [v[-1] for v in loss.values()]; mm = list(roxM.values())
for (k, c, mk, a, b) in zip(loss, cols, "osD", fl, mm):
    axes[1].scatter(a, b, color=c, marker=mk, s=30); axes[1].annotate(k.split(" ")[0], (a, b), xytext=(4, 3), textcoords="offset points", fontsize=8)
axes[1].set_xlabel("final train loss"); axes[1].set_ylabel(r"$\mathcal{R}$Oxf Medium mAP")
axes[1].set_ylim(71.3, 72.4); axes[1].set_xlim(0.07, 0.125)
for (k, c, mk, a) in zip(loss, cols, "osD", fl):
    axes[2].scatter(a, valM[k], color=c, marker=mk, s=30); axes[2].annotate(k.split(" ")[0], (a, valM[k]), xytext=(4, 3), textcoords="offset points", fontsize=8)
axes[2].set_xlabel("final train loss"); axes[2].set_ylabel("rSfM-120k val mAP")
axes[2].set_ylim(92.5, 93.8); axes[2].set_xlim(0.07, 0.125)
fig.tight_layout(); fig.savefig(f"{OUT}/train_loss.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig: descriptor head x whitening x scale (B0 and B1, ROxf Medium) ----
heads = ["GeM", "CLS", "Avg"]
b0 = {("ss", False): [61.27, 64.59, 62.08], ("ss", True): [64.30, 69.28, 65.81],
      ("ms", False): [62.58, 65.41, 62.54], ("ms", True): [65.15, 68.45, 64.86]}
b1 = {("ss", False): [37.26, 70.41, 36.18], ("ss", True): [55.85, 72.10, 54.14],
      ("ms", False): [39.57, 67.79, 38.33], ("ms", True): [55.71, 69.82, 53.81]}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
lab = {("ss", False): "SS", ("ss", True): "SS + $L_w$", ("ms", False): "MS", ("ms", True): "MS + $L_w$"}
pal = ["#9ecae1", "#2a6fb0", "#fdae6b", "#d0662b"]
for ax, d, t in [(axes[0], b0, "B0: trained on GeM head"), (axes[1], b1, "B1: trained on CLS head")]:
    x = np.arange(3); w = 0.2
    for i, (k, c) in enumerate(zip(d, pal)):
        ax.bar(x + (i - 1.5) * w, d[k], w, color=c, label=lab[k])
    ax.set_xticks(x); ax.set_xticklabels(heads); ax.set_title(t); ax.set_ylim(30, 76)
axes[0].set_ylabel(r"$\mathcal{R}$Oxf Medium mAP")
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, ncol=4, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/heads_ablation.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Fig: GeM pooling power p illustration (theory, not results) ----
rng = np.random.default_rng(0); acts = np.sort(rng.gamma(1.2, 1.0, 64))[::-1]
ps = np.linspace(1, 12, 100); g = [np.mean(acts ** p) ** (1 / p) for p in ps]
fig, ax = plt.subplots(figsize=(3.3, 2.2))
ax.plot(ps, g, color="#2a6fb0"); ax.axhline(acts.mean(), ls="--", color="#777", lw=0.9); ax.axhline(acts.max(), ls=":", color="#777", lw=0.9)
ax.text(11.8, acts.mean() + 0.08, "average (SPoC, $p=1$)", ha="right", fontsize=7)
ax.text(11.8, acts.max() - 0.3, r"max (MAC, $p\to\infty$)", ha="right", fontsize=7)
ax.axvline(3, color="#d0662b", lw=0.9); ax.text(3.2, acts.mean() + 0.6, "$p=3$", color="#d0662b", fontsize=7)
ax.set_xlabel("pooling exponent $p$"); ax.set_ylabel("GeM value $f_k$")
fig.tight_layout(); fig.savefig(f"{OUT}/gem_p.pdf", bbox_inches="tight"); plt.close(fig)
print("ok")
