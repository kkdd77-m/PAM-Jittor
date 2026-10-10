# 从 AutoDL 数据盘补齐文件清单

关机/提交前，把下列文件从实例数据盘 `/root/autodl-tmp/PAM/` 下载到本仓库对应位置。
大权重与大数据**不入库**（见 `.gitignore`），仅代码、日志、图片、小 npz 基准入库。

## 必下：代码

| 远程路径 | 放到仓库 |
|---|---|
| `jittor_pam/*.py`（model.py / layers.py / controlnet.py / vae.py / jt_a800.py 等） | `jittor_pam/` |
| `torch_ref/pt_*.py` | `torch_ref/` |
| `PAM/appearance_gen/`（官方源码） | `torch_ref/PAM/appearance_gen/`（或在 README 指引现场下载） |

## 必下：对齐证据（小文件，入库）

| 远程路径 | 放到仓库 |
|---|---|
| `ref/loss_curve_compare.png` | `assets/loss_curve_compare.png` |
| `ref/train_alignment_log.txt` | `ref/train_alignment_log.txt`（或 docs/） |
| `ref/a800_loss_curve.npy` | `ref/a800_loss_curve.npy` |
| `ref/pt_loss_curve.npy` | `ref/pt_loss_curve.npy` |
| `ref/cn_init.npz` | `ref/cn_init.npz` |
| 采样可视化图（Jittor vs PyTorch 拼图） | `assets/` |

## 可选：较大基准（视仓库体积，可放 Release/云盘）

| 远程路径 | 大小 | 说明 |
|---|---|---|
| `ref/train_step_ref.npz` | 约 392MB | 单步梯度基准（dres、代表参数梯度） |
| `jittor_pam/docs/ALIGNMENT_REPORT.md`、`AUTODL_SETUP.md` | 小 | 旧对齐证据，可并入 docs/ |

## 不入库（.gitignore 已排除）

- `weights/**`（flux1-dev.safetensors 约 23GB、ae.safetensors）；
- `ref/train_batch.npz` 及含版权/隐私的原始数据；
- `__pycache__/`、`*.pyc`、Jittor 编译缓存 `~/.cache/jittor`。

## 下载方式（任选）

- AutoDL 的 JupyterLab 文件管理器右键下载；
- 或用 `scp`/`rsync`：

```bash
scp -P <SSH端口> -r root@<主机>:/root/autodl-tmp/PAM/ref/loss_curve_compare.png ./assets/
scp -P <SSH端口> -r root@<主机>:/root/autodl-tmp/PAM/jittor_pam ./
```
