# ref/ 对齐数据目录

运行训练对齐后，本目录应包含（从 AutoDL 数据盘下载或本地运行生成）：

- `cn_init.npz`：PyTorch/Jittor 共享的 ControlNet input_hint 初始化；
- `pt_loss_curve.npy`：PyTorch 20 步 loss 曲线；
- `a800_loss_curve.npy`：Jittor 20 步 loss 曲线；
- `train_alignment_log.txt`：逐点数值与统计（由 `scripts/plot_loss_curve.py` 生成）；
- `train_step_ref.npz`（可选，约 392MB）：全模型单步前向/反向梯度基准。

`train_batch.npz` 等含数据/权重的大文件不入库，见 docs/DOWNLOAD_CHECKLIST.md。
