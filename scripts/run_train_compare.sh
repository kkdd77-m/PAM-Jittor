#!/usr/bin/env bash
# PAM ControlNet 少量数据训练：PyTorch vs Jittor 减层对齐
# 工作根目录：/root/autodl-tmp/PAM
set -e
cd "$(dirname "$0")/.."

N=${1:-20}
LR=${2:-3e-4}
FLT='Compiling|cache_path|Found nv|cuda key|restart|Jittor\(|g\+\+|Load cc|mpicc'

echo "==== [1/2] PyTorch 基准（同时导出 cn_init.npz） ===="
python3 torch_ref/pt_train_curve.py 2>&1 | grep -vE "Warning|warn"

echo "==== [2/2] Jittor 训练（加载共享初始化） ===="
python3 jittor_pam/jt_a800.py "$N" "$LR" 2>&1 | grep -vE "$FLT"

echo "==== 对比 ===="
python3 - "$N" <<'PY'
import sys, numpy as np
jt=np.load('ref/a800_loss_curve.npy'); pt=np.load('ref/pt_loss_curve.npy')
for i,(a,b) in enumerate(zip(jt,pt)):
    print('%4d  %.6f  %.6f  %.2e'%(i,a,b,abs(a-b)))
print('corr %.6f  meanrel %.4f%%  maxabs %.4e'%(
    np.corrcoef(jt,pt)[0,1],
    100*(np.abs(jt-pt)/np.abs(pt)).mean(),
    np.abs(jt-pt).max()))
PY

echo "==== 画图 ===="
python3 scripts/plot_loss_curve.py
