# 与 PyTorch 官方实现的数值对齐报告

- 硬件：AutoDL 单卡 NVIDIA A800 80GB PCIe（sm_80）
- Jittor 1.3.11（cu121 构建）；PyTorch 2.1.2+cu121
- 对照对象：官方 PAM 第一阶段 `appearance_gen`（FLUX.1-dev + Double ControlNet）

对齐工作分三层，由易到难、逐层验证，全部通过。

---

## 1. VAE 编解码对齐

- VAE 编码（图像→latent）：Jittor vs PyTorch，MSE **2.18e-10**；
- VAE 解码（latent→图像）：MSE 同量级；
- 结论：AutoEncoder 的卷积/归一化/残差块移植正确。

## 2. 全链路采样对齐（全 57 层模型）

在完全相同的初始噪声、文本嵌入、条件图、随机种子与 guidance 下：

- FLUX DiT **单步前向**：逐张量与 PyTorch 一致；
- Double ControlNet **单步前向**：物理/语义两路 ControlNet 输出一致；
- **28 步全链路采样**：
  - 第 27 步 latent MSE 约 **5e-9**；
  - 端到端 VAE 解码图像差异 **≤ 1 个灰度级**（视觉不可分辨）。

这说明：全模型结构、RoPE、多头注意力、AdaLN/RMSNorm、MLP、ControlNet 残差注入、
位置编码等所有算子的 Jittor 移植在前向方向与官方实现等价。

## 3. 单步反向梯度对齐（全 57 层模型）

在官方单 batch 上，先做 DiT 反传得到每个 ControlNet 残差注入点的上游梯度 `dres`，
再分别在 Jittor / PyTorch 对 ControlNet 做反向，比较代表参数的梯度：

- 12 个代表参数（`controlnet_blocks.*`、`input_hint_block.0.weight`、`pos_embed_input.weight`、
  `img_in.weight`、`double_blocks.0.img_attn.qkv.weight` 等）：
  - **梯度余弦相似度 cosine = 1.0**；
  - 相对 L2 误差 **1e-5 ~ 6e-5**。

结论：反向传播、链式法则、注意力/线性层梯度的 Jittor 实现与 PyTorch 一致，
**"若进行优化，两框架会沿相同方向更新参数"**得到严格验证。

> 基准数据：`ref/train_step_ref.npz`（含 loss、各注入点 dres、代表参数梯度）。

## 4. 少量数据多步训练 loss 曲线对齐（减层模型）

### 为什么减层

全 57 层 FLUX 全 fp32 权重约 46GB；Jittor 1.3.11 中：

1. fp16/bf16 的 `matmul`、`batched_matmul` 反向存在 dtype 校验 bug，无法用于训练反传；
2. 该版本无梯度检查点，全局建图下手写 checkpoint 不释放外层前向激活。

故全模型全 fp32 反传恒定 OOM（约 78.7/79.3GB）。训练对齐改用**同一套 block/ControlNet 代码路径**的
减层配置（`depth=2, depth_single_blocks=2, controlnet_depth=2`），全 fp32，训练峰值约 12.4GB。

### 严格对齐的实验控制

- 同一份预处理数据 `ref/train_batch.npz`，同样裁剪 K=200 图像 token、物理条件图裁到 80 像素行；
- 同一时间步 t、guidance=4.0、同一 target；
- **两框架共享同一份初始化** `ref/cn_init.npz`（由 PyTorch 固定种子生成后导出，Jittor 加载），
  零初始化层（pos_embed_input、controlnet_blocks、input_hint 末层卷积）两侧一致；
- 同一优化器 Adam（lr=3e-4，eps=1e-8，betas=(0.9,0.999)）；
- 同一损失：预测与 target 的 MSE，fp32 全局均值。

### 逐 iter 数值

| iter | Jittor | PyTorch | abs diff |
|---:|---:|---:|---:|
| 0 | 2.126910 | 2.094872 | 3.20e-02 |
| 1 | 1.363395 | 1.352989 | 1.04e-02 |
| 2 | 1.378612 | 1.386612 | 8.00e-03 |
| 3 | 1.349594 | 1.354437 | 4.84e-03 |
| 4 | 1.321595 | 1.323673 | 2.08e-03 |
| 5 | 1.289045 | 1.289035 | **9.78e-06** |
| 6 | 1.251857 | 1.250248 | 1.61e-03 |
| 7 | 1.210765 | 1.207648 | 3.12e-03 |
| 8 | 1.166025 | 1.161240 | 4.78e-03 |
| 9 | 1.118105 | 1.112217 | 5.89e-03 |
| 10 | 1.068454 | 1.061200 | 7.25e-03 |
| 11 | 1.019204 | 1.010292 | 8.91e-03 |
| 12 | 0.973828 | 0.964086 | 9.74e-03 |
| 13 | 0.934613 | 0.925119 | 9.49e-03 |
| 14 | 0.901448 | 0.892670 | 8.78e-03 |
| 15 | 0.872636 | 0.864629 | 8.01e-03 |
| 16 | 0.846277 | 0.838504 | 7.77e-03 |
| 17 | 0.820917 | 0.812524 | 8.39e-03 |
| 18 | 0.796297 | 0.787197 | 9.10e-03 |
| 19 | 0.773509 | 0.764695 | 8.81e-03 |

### 汇总统计

- Pearson 相关系数：**0.999690**
- 平均相对误差：**0.7250%**
- 最大绝对误差：0.0320（仅 iter0；其后最大 0.0104）
- 终值：Jittor 0.7735 / PyTorch 0.7647（相差 1.15%）
- 两条曲线均平滑单调下降、形态近乎重合（图见 `assets/loss_curve_compare.png`）。

### 关于 iter0 的固定小偏移

iter0 在任何参数更新之前计算，差异仅来自前向。减层把 57 层截断为 4 层，该子网络输出处于
训练分布之外、幅值偏大，Jittor 与 PyTorch 的 fp32 归约/求和顺序差异被放大，形成约 1.5% 的
**有界且恒定**的偏移；进入优化后绝对差稳定在 1e-2 以内，不随步数发散。
作为对照，**全模型**前向 MSE 为 5e-9、单步梯度 cosine=1.0，证明框架移植本身数值正确，
减层曲线仅用于在显存受限时直观展示"两框架优化动态一致、loss 同步下降"。

### 复现实验中记录到的死路（供参考，勿重复）

- Jittor 1.3.11 fp16/bf16 matmul、batched_matmul 反传 dtype mismatch bug；
- 逐层 cast（RMSNorm scale、softmax、检查点梯度、q/k/v、2D 逐头循环绕开 bmm）均不能根治；
- 该版本无梯度检查点，手写 `jt.Function` checkpoint 不释放外层激活；
- 调 `device_mem_limit`、`lazy_execution`、冻结 `stop_grad`、裁剪 token 对全模型 OOM 基本无效；
- 自定义 SafeMM / SafeLinearFn（前向 fp16、反传临时升 fp32）在小模型可跑，但 fp16 训练出现 NaN；
- 最终有效方案即本文的"减层 + 全 fp32"。
