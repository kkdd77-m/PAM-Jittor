#!/usr/bin/env python3
"""读取 PyTorch / Jittor 两条训练 loss 曲线，生成对照图与对齐日志。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "ref")
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

jt = np.load(os.path.join(REF, "a800_loss_curve.npy"))
pt = np.load(os.path.join(REF, "pt_loss_curve.npy"))
it = np.arange(len(jt))
corr = np.corrcoef(jt, pt)[0, 1]
mr = (np.abs(jt - pt) / np.abs(pt)).mean()

plt.figure(figsize=(8, 5.2))
plt.plot(it, pt, "-o", color="#1f6feb", ms=5, lw=2, label="PyTorch (official)")
plt.plot(it, jt, "--s", color="#d1242f", ms=5, lw=2, label="Jittor (ours)")
plt.xlabel("Training iteration")
plt.ylabel("MSE loss")
plt.title("PAM Double-ControlNet training alignment\n"
          "(reduced Flux depth=2/single=2, fp32, Adam lr=3e-4, 200 tokens)")
plt.grid(alpha=.3)
plt.legend()
plt.text(0.30, 0.55,
         "Pearson corr = %.5f\nmean rel.err = %.3f%%\nfinal loss %.4f vs %.4f"
         % (corr, 100 * mr, jt[-1], pt[-1]),
         transform=plt.gca().transAxes, fontsize=11,
         bbox=dict(boxstyle="round", fc="#f6f8fa", ec="#999"))
plt.tight_layout()
out_png = os.path.join(ASSETS, "loss_curve_compare.png")
plt.savefig(out_png, dpi=160)

with open(os.path.join(REF, "train_alignment_log.txt"), "w") as f:
    f.write("PAM ControlNet Jittor vs PyTorch training alignment (AutoDL A800-80G)\n")
    f.write("config: Flux depth=2/single=2, controlnet_depth=2, cond_ch=6, fp32, "
            "tokens K=200, Adam lr=3e-4 eps=1e-8, 20 iters, shared init cn_init.npz\n\n")
    f.write("iter  jittor      pytorch     absdiff\n")
    for i, (a, b) in enumerate(zip(jt, pt)):
        f.write("%4d  %.6f  %.6f  %.2e\n" % (i, a, b, abs(a - b)))
    f.write("\nPearson corr = %.6f\nmean rel err = %.4f%%\nmax abs diff = %.4f\n"
            "final: jittor %.4f / pytorch %.4f\n"
            % (corr, 100 * mr, np.abs(jt - pt).max(), jt[-1], pt[-1]))

print("saved", out_png)
print("corr %.6f meanrel %.4f%% final %.4f/%.4f" % (corr, 100 * mr, jt[-1], pt[-1]))
