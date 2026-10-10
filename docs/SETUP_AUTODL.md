# AutoDL A800 环境搭建记录

## 1. 租用实例

- 镜像：Ubuntu 22.04 + CUDA 12.1 + Python 3.10（Jittor 以 cu121 构建）。
- GPU：单卡 A800 80GB PCIe 即可；减层训练/推理在 ≥16GB 卡上也能跑。
- 数据放在**数据盘** `/root/autodl-tmp`（IO 快、关机保留，但不随镜像保存）；
  代码如需随镜像保留可放系统盘 `/`。

## 2. 每次进 shell

```bash
cd /root/autodl-tmp/PAM
# 若报 libstdc++ / GLIBCXX 相关错误：
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

> 重启后默认在 `~`，需要重新 `cd`。

## 3. 安装依赖

```bash
python3 -m pip install jittor==1.3.11
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121   # 仅对照基准
pip install einops accelerate safetensors matplotlib numpy
```

## 4. 获取官方 PAM 第一阶段源码

`git clone` 在部分区域网络会超时，改用 codeload zip（给足超时，约 8.7MB）：

```bash
curl -L --retry 3 https://codeload.github.com/GasaiYU/PAM/zip/refs/heads/main -o /tmp/PAM.zip
cd /tmp && unzip PAM.zip
mkdir -p /root/autodl-tmp/PAM/PAM
cp -r /tmp/PAM-main/appearance_gen /root/autodl-tmp/PAM/PAM/
```

对照脚本通过以下路径 import：

```python
sys.path.insert(0,'/data/PAM/PAM/appearance_gen')
sys.path.insert(0,'/data/PAM/PAM/appearance_gen/src')
```

其中 `/data/PAM` 是指向 `/root/autodl-tmp/PAM` 的软链接。

## 5. 权重与数据目录

```
/root/autodl-tmp/PAM/
├── weights/flux_dev/flux1-dev.safetensors   # FLUX.1-dev bf16，约 23GB
├── weights/flux_dev/ae.safetensors          # VAE，约 320MB
├── ref/train_batch.npz                       # 单 batch 预处理数据
├── jittor_pam/                               # Jittor 实现
└── torch_ref/                                # PyTorch 对照脚本
```

## 6. 过滤编译日志

Jittor 首次编译输出较多，可用：

```bash
python3 jittor_pam/jt_a800.py 20 3e-4 2>&1 \
  | grep -vE "Compiling|cache_path|Found nv|cuda key|restart|Jittor\(|g\+\+|Load cc|mpicc"
```

## 7. 省钱要点

- **用完立刻在控制台「关机」（不是释放）**：关机即停 GPU 计费，数据盘保留；
- README/PPT/视频等不需要 GPU 的工作请在本地完成；
- 长任务用 `screen`/`tmux` 挂后台，避免 SSH 断开中断。
