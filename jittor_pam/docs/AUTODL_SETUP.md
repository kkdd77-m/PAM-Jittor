# PAM → Jittor 迁移：AutoDL 复现与续作指南

## 一、当前进度（第一阶段 Appearance Generation）
- 已完成并数值对齐：VAE 解码器 / VAE 编码器 / 双 ControlNet 单步 / FLUX DiT 单步 / 28 步全链路出图 /
  训练数据预处理 / ControlNet 反向梯度（cosine=1.0）。
- 唯一未在旧机(V100-32GB)完成：多步优化器 loss 下降（显存不足，非正确性问题）。
- 详见 docs/ALIGNMENT_REPORT.md。

## 二、新机配置建议
- GPU：**≥40GB 显存**（A100/A800 40/80GB；第二阶段视频模型建议直接 80GB）。V100/3090 等 32GB 仅能跑推理对齐。
- 镜像：Ubuntu + CUDA 12.x + Python 3.10。
- 环境：
  - Jittor：`pip install jittor==1.3.11.0`（首次运行会在 ~/.cache/jittor 自动下载 cuda12.2+cudnn8 工具链并编译，勿迁移该缓存，让它重建）。
  - PyTorch 参考环境：venv 装 torch==2.7.1（cu126 轮子，自带运行时，不依赖系统 CUDA）、safetensors、einops、decord、diffusers、accelerate、transformers、omegaconf。

## 三、目录与权重（权重大，新机重新下载，勿打包迁移）
期望布局：

/data/PAM/
PAM/                     # 官方仓库：git clone [https://github.com/GasaiYU/PAM](https://github.com/GasaiYU/PAM)
jittor_pam/              # 本迁移代码（见包内）
ref/                     # 金标准参考数据（见包内 ref_data，放到这里）
weights/
flux_dev/flux1-dev.safetensors   (~24GB, bf16)
flux_dev/ae.safetensors          (VAE)
pam_ckpt/appearance_gen/controlnet.bin  (2.8GB, bf16, 双 CN 源)
xflux_text_encoders/             (T5, 8.9G)
clip_vit_l14/                    (CLIP, 6.4G)
权重下载脚本参考官方 weights/download_weights.py。
CN 的 npz（fp16 推理用 / bf16u16 梯度对齐用）不放包，新机执行：
`/path/torchvenv/bin/python /data/PAM/jittor_pam/export_cn_npz.py` 一键再生。

## 四、放置本包内容
- jittor_pam/ → /data/PAM/jittor_pam/
- ref_data/* → /data/PAM/ref/
- torch_ref/pt_*.py → 复制进 /data/PAM/PAM/appearance_gen/（它们 import src.flux，必须在该目录运行）
- comb_val/ → /data/PAM/PAM/val_resources/comb_val/；dexycb_prompt.jsonl → /data/PAM/PAM/

## 五、复现验证（Jittor 用系统 python3，PyTorch 用 torchvenv）
1. VAE 解码：`python3 jt_vae_test.py`
2. 双 CN 单步：`python3 jt_cn_test.py`（需 controlnet fp32，见 pt_cn_save.py；或改用 fp16）
3. DiT 单步：`python3 jt_mixed.py`
4. 全链路：`python3 jt_full_loop.py`（期望 step27 MSE~5e-9）→ `jt_decode.py` → `compare_final.py`（图像差≤1灰度级）
5. 数据/编码：torch `pt_train_batch.py` 后 `python3 jt_train_batch.py`（latent MSE~2e-10）
6. 梯度：torch `pt_train_step.py` 后（先 export_cn_npz）`python3 jt_train_step.py`（cosine=1.0）

## 六、第一阶段最后一步（≥40GB 卡）
跑 `python3 jt_train_descent.py 100 1e-3`（全参数 AdamW，固定单样本过拟合），
观察 loss 从 ~0.33 持续下降即达成"小数据 loss 下降"判据；随后更新 ALIGNMENT_REPORT，第一阶段闭环。
显存若仍紧张，按报告里的 jt.Function 梯度检查点 / 两阶段方案处理。

## 七、第二、三阶段续作
- 第二阶段 CogVideoX I2V + SpaTracker：读 PAM/training/cogvideox_image_to_video_comb_sft.py、
  PAM/models/spatracker；视频模型更吃显存，沿用"torch 存参考→逐层 jittor 复现→数值对比"方法。
- 第三阶段 GraspXL。
- 真实训练数据（dexycb/oakink）本机没有，需另行下载并预留磁盘；当前仅有 comb_val 演示帧。
