# PAM → Jittor 迁移：第一阶段 Appearance Generation 对齐报告

> 目标：把 PAM（Pose-Appearance-Motion，CVPR 2026）的 PyTorch 实现迁移到 Jittor 并数值复现。
> 硬件：阿里云 ECS，V100-SXM2-32GB（sm_70，**不支持 bf16**），12 vCPU / 89.85GB RAM / 无 swap，磁盘 99GB。
> 环境：系统 `python3` = Jittor 1.3.11.0；`/opt/torchvenv/bin/python` = PyTorch 2.7.1+cu126（读 bf16、生成参考）。

## 一、对齐结果（截至 2026-09-19）

| 梯队 | 模块 | 金标准脚本(Jittor) | PyTorch参考 | 指标 | 判定 |
|---|---|---|---|---|---|
| 1 | VAE Decoder | `jt_vae_test.py` | ref.png | 图像 MSE 0.015(0-255) | ✅ |
| 2 | 双 ControlNet 单步 | `jt_cn_test.py` | `pt_cn_save.py`→cn_out.npz | MSE ~1e-11 | ✅ |
| 3 | FLUX DiT 单步 | `jt_mixed.py` | `pt_main_step.py`→main_pred.npy | MSE 4.22e-10 / MAX 1.37e-4 | ✅ |
| 4a | 28 步全链路 latent | `jt_full_loop.py` | `pt_full_loop.py`→loop_latents.npz | step27 MSE 4.99e-9 / MAX 2.28e-3 | ✅ |
| 4b | VAE 端到端出图 | `jt_decode.py`+`compare_final.py` | `pt_decode.py` | 解码张量 MSE 2.42e-8；图像最大差 1/255 灰度级，RMSE 0.127 | ✅ |

误差随步数缓慢线性累积（正常），最终图像与 PyTorch 近乎逐像素一致。**第一阶段推理链路（Step 1-4）全部对齐。**

## 二、采样管线（官方 denoise_double_control 的复刻）

- 输入：seed=42 的初始噪声 packed `[1,1200,64]`；T5/CLIP 文本嵌入；4 张像素空间条件图 480x640。
- 条件：cn1 吃 physical=cat(depth,normal)（6通道）；cn2 吃 semantic=cat(hand,seg)（6通道）。
- 每步：r1=cn1(...)，r2=cn2(...)；残差 res = r1*gs0 + r2*gs1（gs=(1,1,0,0)，各 2 个残差）；
  主模型在 19 个 double_blocks 内按 `img += res[index_block % 2]` 循环注入。
- 步进：Euler `img += (t_prev - t_curr) * pred`，共 28 步；schedule=linspace(1,0,29) 经 time_shift。
- 收尾：unpack 到 `[1,16,60,80]` → VAE decode → clamp[-1,1] → 图。
- `inp_condition_default` 在官方代码里**未使用**（相关调用全注释），无需实现。

## 三、显存跑通配方（关键，务必沿用）

V100 32GB 上 Flux-dev ~12B（bf16 权重 24GB），最终稳定方案：
1. **tiny init**：monkey-patch `jt.nn.Linear.__init__` 只建 1x1 零权重，避免建模型时分配 ~48GB 随机 fp32 权重。
2. **fp16 存权重 / fp32 计算**：主模型权重读成 fp16 上 GPU，patch `Linear.execute` 内部 `w=self.weight.float32()` 临时升精度，输入/激活全 fp32（纯 fp16 计算会 NaN，V100 无 bf16）。
3. **手动逐 block 前向**：不调 model()，自己写 double/single block 循环，每个 block 后 `jt.clean_graph(); jt.gc()`。
4. `jt.flags.device_mem_limit` 设上限（单步 28GB，全链路 31GB）逼 Jittor 主动回收；`lazy_execution=0`。
5. 双 ControlNet 用 **fp16 权重**（`controlnet_fp16.npz`，Conv 层 input_hint_block 转 fp32），CN 不加载用不到的 single_blocks；cn1 与 cn2 之间各 clean_graph+gc。

显存：主模型加载 23.0GB，加双 CN 后 26.5GB，全链路峰值 ~31GB。

## 四、已验证的死路（勿重复）

- GPU fp16 全精度直接加载：权重 23GB + 内存池/激活 → 31.3GB OOM。
- GPU int8（314 个 Linear per-channel）：仍 OOM，且反量化 MSE 0.139/MAX 1.81，精度差，弃用（`archive/dead_ends/jt_win3.py`）。
- 纯 fp16 计算：NaN（V100 无 bf16，fp16 溢出）。
- CPU fp32 / CPU int8：被 Linux OOM Killer 杀（Exit 137）。根因：Jittor CPU 内存池用 mmap 共享内存预分配，dmesg 中 shmem-rss 高达 86-90GB。
- `JT_CPU_MEM_LIMIT` 环境变量无效；`jt.flags.cpu_mem_limit` import 后设置太晚。
- 混合 dtype：GPU cublas batched matmul 要求两输入同 dtype；attention softmax 输出 fp32 与 fp16 的 v 相乘报错（已在 layers.py 加 cast）。

## 五、Jittor 踩坑备忘

- numpy safetensors backend 读不了 bf16；手动：`(u16.astype(np.uint32)<<16).view(np.float32)`，u16 dtype '<u2'。
- Var 无 `.T`，用 `.transpose()`；2D matmul 支持混合 dtype，3D batched 不行。
- `jt.sync_all()` 在 Module 内报 "self=module"，统一改用 `jt.gc()`。
- `load_state_dict` 不支持 strict；形状不匹配会跳过并打印 "load parameter ... failed"（不是报错退出，要盯日志）。
- tiny-init 是全局 patch：之后再建 ControlNet 需先恢复原 `Linear.__init__`，或直接 setattr 灌权重。
- Jittor restart 会丢 heredoc stdin：脚本先落盘再跑；写完用 `python3 -c "import ast; ast.parse(...)"` 校验。
- 长任务 nohup + 日志；`ps aux|grep` 看存活，`dmesg|tail` 看 OOM，`nvidia-smi` 看显存。
- torch 脚本 VAE 卷积报 "GET was unable to find an engine"：开头加 `torch.backends.cudnn.enabled=False` 并用 autocast(fp32)。

## 六、文件清单

核心实现（/data/PAM/jittor_pam/）：
- model.py(Flux) / layers.py(attention/RoPE/emb) / controlnet.py(ControlNetFlux) / autoencoder.py(VAE)

金标准脚本（同目录）：
- jt_vae_test.py, jt_cn_test.py, jt_mixed.py（三个单步）
- jt_full_loop.py（28步全链路）, jt_decode.py（VAE解码）, compare_final.py（出图对比）

PyTorch 参考（/data/PAM/PAM/appearance_gen/）：
- pt_gen_ref.py(文本/噪声) pt_cn_save.py(CN权重/条件) pt_main_step.py(DiT单步)
- pt_full_loop.py(28步) pt_decode.py(VAE解码)；用 /opt/torchvenv/bin/python 跑

参考产物（/data/PAM/ref/）：
- inp_true.npz, conds.npz, schedule.npy, main_pred.npy, cn_out.npz
- loop_latents.npz(step_0..27), final_packed/final_unpacked.npy（torch）
- jt_final_packed/jt_final_unpacked.npy, pt_decoded.npy, jt_decoded.npy
- final_compare.png(左torch/右jittor), final_diff_x10/x50.png
- controlnet_fp16.npz(2.8G,生产用)；fp32 版(5.6G)已删，需要时跑 pt_cn_save.py 再生

权重（/data/PAM/weights/）：flux_dev/(主+ae,23G)、pam_ckpt/appearance_gen/controlnet.bin(2.8G,bf16源)、
  xflux_text_encoders(T5)、clip_vit_l14。

## 七、下一步（第一阶段剩余 + 后续阶段）

- Step 5：数据加载对齐（复刻 HOIVideoDatasetResining：读视频帧、4 类条件、latent 缓存），对齐单个 batch。
- Step 6：训练循环 + loss，验证小数据上 loss 下降（第一阶段完成标志）。
- 第二阶段：CogVideoX I2V + SpaTracker 运动生成。第三阶段：GraspXL 姿态对接。

---

# 第二部分：Step 5-6 数据与训练对齐（2026-09-19 续）

## Step 5 数据加载/预处理对齐 ✅
- 机器上无真实训练数据（无 dexycb/oakink、无 label npz/列文件），用 val_resources/comb_val 单帧构造最小固定 batch（torch 存参考→jittor 复现）。
- 训练 flow（复刻 double_controlnet.py）：x_1=VAE.encode(干净帧, 强制 reg.sample=False 取均值)→pack；
  x_0=固定种子高斯噪声；t=0.5；x_t=(1-t)x_1+t x_0；target=x_0-x_1；
  physical=cat(depth,normal)，semantic=cat(hand,seg)；guidance=4。
- PyTorch 参考 pt_train_batch.py 存 /data/PAM/ref/train_batch.npz；Jittor jt_train_batch.py 复现：
  - **VAE 编码器 latent MSE 2.18e-10 / MAX 1.34e-3**（首次验证 encoder，此前只验过 decoder）
  - pack / x_t / target MSE 均 ~1e-10。数据进入模型前所有张量对齐。

## Step 6 训练步对齐
- 训练只更新两个 ControlNet（DiT/VAE/T5/CLIP 全冻结），官方未冻结 CN 任何参数（含从 DiT 拷贝的 2 个 double_blocks），每 CN 约 7 亿参数。
- PyTorch 单训练步参考 pt_train_step.py（两阶段+梯度检查点，峰值 30.2GB）：
  阶段A 删 CN、DiT 逐 block checkpoint 反传拿残差梯度 dL/dres；阶段B 删 DiT、fp32 CN 用 dres 反传拿参数梯度。loss=0.3341，存 train_step_ref.npz。
- Jittor jt_train_step.py（CN bf16→fp32 精确权重，controlnet_bf16u16.npz）对比 12 个代表性参数梯度
  （含 3 亿维 double_blocks.0.img_attn.qkv.weight）：**cosine 全部 = 1.0000000，相对 L2 误差 1e-5~6e-5**。
  即 ControlNet（唯一可训练部分）的反向传播数学在 Jittor 完全正确。

## Step 6 多步 loss 下降：受 32GB 显存阻塞（非正确性问题）
- jt_train_descent.py（全参数 Adam，DiT 用 jt.Function 手写梯度检查点）在 V100-32GB 上 OOM：
  DiT fp16 常驻 23GB + 双 CN + 全分辨率(480x640)条件 CNN 反向激活 + 优化器状态超过 32GB。
- jt_train_descent_lite.py 为 32GB 精简版（预计算条件 CNN、只训适配器 controlnet_blocks/pos_embed、SGD），仍在 31.3GB 临界 OOM。
- **结论**：前向、编码、反向梯度均已数值对齐；多步迭代下降仅差显存容量。在 ≥40GB GPU（A100/A800 40/80GB）上
  直接跑 jt_train_descent.py（全参数 AdamW）即可完成"小数据 loss 下降"这一第一阶段收尾判据。
