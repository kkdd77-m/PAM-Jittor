# 性能 Log

测试环境：AutoDL GPU 云主机

| 项 | 配置 |
|---|---|
| GPU | 1 × NVIDIA A800 80GB PCIe（sm_80） |
| 驱动 / CUDA | 580.105.08 / CUDA 13.0 宿主；Jittor 以 cu121 编译 |
| CPU / 内存 | 14 核 / 120 GB |
| 系统盘 / 数据盘 | 30 GB / 107 GB |
| Jittor / PyTorch | 1.3.11 / 2.1.2+cu121 |

## 显存占用

| 阶段 | 精度 | 峰值显存 |
|---|---|---|
| 全 57 层 FLUX 权重常驻（推理/采样） | bf16 | 约 23 GB（`dit loaded 23151 MiB`） |
| 全 57 层 FLUX 权重常驻（推理/采样） | fp32 | 约 45.9 GB（`dit loaded 45857 MiB`） |
| 减层（depth=2/single=2）训练前向+反向 | fp32 | **约 12.4 GB**（`mem ≈ 12403 MiB`） |
| 减层模型权重常驻 | fp32 | 约 2.4 GB（`dit loaded 2411 MiB`） |
| 全 57 层 fp32 训练反传 | fp32 | OOM（约 78.7 / 79.25 GB） |

> 说明：全模型 fp32 反传 OOM 的根因是 Jittor 1.3.11 缺少梯度检查点、且 fp16 反传有 dtype bug，
> 详见 [ALIGNMENT.md](ALIGNMENT.md) 第 4 节。这是框架版本限制，而非算法/实现问题。

## 训练耗时（减层，20 步，K=200 token）

- 首次运行包含 Jittor 算子即时编译（JIT compile），耗时较长；
- 编译完成后每步前向+反向为秒级（A800），20 步总训练耗时约 1 分钟量级（不含权重加载与编译）；
- PyTorch 对照（同样减层、fp32）量级相同。

> 如需在 README/视频中给出精确秒数，可在两脚本中补充 `time` 计时后回填本表。

## 可训练参数量

- ControlNet 新增可训练参数（input_hint_block + pos_embed_input + controlnet_blocks）：
  约 **19.1 M**（两框架一致）。

## 复现实验的稳定性观察

- fp32 减层训练：loss 单调下降、无 NaN，结果可重复（共享初始化）；
- fp16 减层训练：在该 Jittor 版本出现 NaN（matmul 反传 bug 所致），不推荐；
- 每次进 shell 需 `cd /root/autodl-tmp/PAM` 并在必要时
  `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`。
