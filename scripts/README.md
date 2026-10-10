# 一键脚本说明

> 脚本假设工作根目录为 `/root/autodl-tmp/PAM`（即 `/data/PAM`），在 AutoDL 数据盘内运行。

- `run_train_compare.sh`：依次运行 PyTorch 基准与 Jittor 减层训练，产出两条 loss 曲线，
  并打印逐点对比与相关系数/相对误差。
- `plot_loss_curve.py`：读取两条 `.npy` 曲线，生成 `assets/loss_curve_compare.png` 与
  `ref/train_alignment_log.txt`。
- `run_sample.sh`：Jittor 与 PyTorch 的 28 步采样对齐（脚本名按数据盘实际文件回填）。

## 训练对齐一条命令

```bash
bash scripts/run_train_compare.sh
```

## 仅重新画图

```bash
python3 scripts/plot_loss_curve.py
```
