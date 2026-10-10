import math
import numpy as np
import jittor as jt

# ===== 自定义算子：前向 fp16，反传临时 fp32（绕开 jittor fp16 matmul 反传 bug）=====
class SafeMM(jt.Function):
    def execute(self, a, b):
        self.a = a; self.b = b
        return jt.matmul(a, b)
    def grad(self, g):
        def Tt(x):
            ax = list(range(x.ndim)); ax[-1], ax[-2] = ax[-2], ax[-1]
            return x.transpose(*ax)
        a = self.a.float32(); b = self.b.float32(); gg = g.float32()
        ga = jt.matmul(gg, Tt(b)).float16()
        gb = jt.matmul(Tt(a), gg).float16()
        return ga, gb

class SafeLinearFn(jt.Function):
    def execute(self, x, w, b, train):
        self.w = w; self.train = train
        if train: self.x = x
        return jt.matmul(x, w.transpose(1, 0)) + b
    def grad(self, gy):
        w = self.w.float32(); g = gy.float32()
        gx = jt.matmul(g, w).float16()
        if self.train:
            x = self.x.float32(); O = g.shape[-1]; K = x.shape[-1]
            g2 = g.reshape(-1, O); x2 = x.reshape(-1, K)
            gw = jt.matmul(g2.transpose(1, 0), x2).float16()
            gb = g2.sum(0).float16()
            return gx, gw, gb
        return gx, jt.zeros(self.w.shape).float16(), jt.zeros((self.w.shape[0],)).float16()

# ============================================================
# 基础工具函数
# ============================================================

def silu(x):
    """SiLU 激活：x * sigmoid(x)"""
    return x * jt.sigmoid(x)

def gelu_tanh(x):
    """GELU tanh 近似（与 PyTorch GELU(approximate='tanh') 完全一致）"""
    return 0.5 * x * (1 + jt.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3))))

class GELU_tanh(jt.Module):
    def execute(self, x):
        return gelu_tanh(x)

# ============================================================
# RoPE 旋转位置编码（对应 math.py 的 rope / apply_rope）
# ============================================================

def rope(pos, dim, theta):
    """
    生成旋转位置编码矩阵。
    pos: [..., seq] 位置坐标
    dim: int（必须为偶数）
    返回: [..., seq, dim//2, 2, 2] 旋转矩阵
    """
    assert dim % 2 == 0
    scale = jt.arange(0, dim, 2, dtype=jt.float64) / dim
    omega = 1.0 / (float(theta) ** scale)
    # 等价于 einsum("...n,d->...nd")
    out = pos.float64().unsqueeze(-1) * omega
    cos = jt.cos(out)
    sin = jt.sin(out)
    # stack [cos, -sin, sin, cos] -> [..., 4]
    out = jt.concat([cos.unsqueeze(-1), (-sin).unsqueeze(-1),
                     sin.unsqueeze(-1), cos.unsqueeze(-1)], dim=-1)
    # reshape [..., 4] -> [..., 2, 2]
    out = out.reshape(list(out.shape[:-1]) + [2, 2])
    return out.float32()

def apply_rope(xq, xk, freqs_cis):
    """
    对 q 和 k 应用 RoPE 旋转。
    xq, xk: [B, H, L, D]
    freqs_cis: [B, 1, L, D//2, 2, 2]
    """
    B, H, L, D = xq.shape
    xq_ = xq.float32().reshape(B, H, L, D // 2, 1, 2)
    xk_ = xk.float32().reshape(B, H, L, D // 2, 1, 2)
    # freqs_cis[...,0] = 第一列[cos, sin], freqs_cis[...,1] = 第二列[-sin, cos]
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(B, H, L, D), xk_out.reshape(B, H, L, D)

def attention(q, k, v, pe):
    q, k = apply_rope(q, k, pe)
    q=q.float32(); k=k.float32(); v=v.float32()
    scale = 1.0 / (q.shape[-1] ** 0.5)
    attn = jt.matmul(q, k.transpose(0, 1, 3, 2)) * scale
    attn = jt.nn.softmax(attn, dim=-1)
    x = jt.matmul(attn, v)
    B, H, L, D = x.shape
    x = x.transpose(0, 2, 1, 3).reshape(B, L, H * D)
    return x
# ============================================================
# 时间步编码 & 位置编码
# ============================================================

def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    """正弦时间步编码（与 PyTorch 版逐行对应）"""
    t = time_factor * t
    half = dim // 2
    # 关键：Python float 提前转 float32，避免 Jittor 提升到 float64
    neg_log = np.float32(-math.log(max_period))
    freqs = jt.exp(neg_log * jt.arange(0, half).astype(jt.float32) / half)
    args = t.float32().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = jt.concat([jt.cos(args), jt.sin(args)], dim=-1)
    if dim % 2:
        embedding = jt.concat([embedding, jt.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class EmbedND(jt.Module):
    """多维位置编码（Flux 用 2D：h轴 + w轴）"""
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def execute(self, ids):
        n_axes = ids.shape[-1]
        emb = jt.concat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

class MLPEmbedder(jt.Module):
    """timestep -> vec 的 MLP"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = jt.nn.Linear(in_dim, hidden_dim, bias=True)
        self.out_layer = jt.nn.Linear(hidden_dim, hidden_dim, bias=True)

    def execute(self, x):
        return self.out_layer(silu(self.in_layer(x)))

# ============================================================
# 归一化层
# ============================================================

class LayerNormNoAffine(jt.Module):
    """无学习参数的 LayerNorm（对应 elementwise_affine=False）"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def execute(self, x):
        mean = x.mean(-1, keepdims=True)
        var = x.pow(2).mean(-1, keepdims=True) - mean.pow(2)
        return (x - mean) / jt.sqrt(var + self.eps)

class RMSNorm(jt.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = jt.ones(dim)

    def execute(self, x):
        x_dtype = x.dtype
        x = x.float32()
        rrms = 1.0 / jt.sqrt(x.pow(2).mean(-1, keepdims=True) + 1e-6)
        return ((x * rrms).astype(x_dtype) * self.scale.cast(x_dtype))

class QKNorm(jt.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def execute(self, q, k, v):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k

# ============================================================
# AdaLN 调制
# ============================================================

class Modulation(jt.Module):
    """
    AdaLN 调制：输入 vec，输出 (shift, scale, gate)。
    double=True 时输出两组（图像流 + 文本流各一组）。
    返回: ((shift1,scale1,gate1), (shift2,scale2,gate2) or None)
    """
    def __init__(self, dim, double):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = jt.nn.Linear(dim, self.multiplier * dim, bias=True)
        self.dim = dim

    def execute(self, vec):
        out = self.lin(silu(vec))[:, None, :]  # [B, 1, multiplier*dim]
        d = self.dim
        if self.is_double:
            return (
                (out[:, :, 0*d:1*d], out[:, :, 1*d:2*d], out[:, :, 2*d:3*d]),
                (out[:, :, 3*d:4*d], out[:, :, 4*d:5*d], out[:, :, 5*d:6*d]),
            )
        else:
            return (out[:, :, 0*d:1*d], out[:, :, 1*d:2*d], out[:, :, 2*d:3*d]), None

# ============================================================
# 自注意力（仅存层，前向逻辑内联在 Block 中）
# ============================================================

class SelfAttention(jt.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = jt.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = jt.nn.Linear(dim, dim)

# ============================================================
# 双流 DiT Block（Flux 核心，内联了 DoubleStreamBlockProcessor 逻辑）
# ============================================================

class DoubleStreamBlock(jt.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, qkv_bias=False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        # 图像流
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = LayerNormNoAffine(hidden_size, eps=1e-6)
        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.img_norm2 = LayerNormNoAffine(hidden_size, eps=1e-6)
        self.img_mlp = jt.nn.Sequential(
            jt.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU_tanh(),
            jt.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # 文本流
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = LayerNormNoAffine(hidden_size, eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.txt_norm2 = LayerNormNoAffine(hidden_size, eps=1e-6)
        self.txt_mlp = jt.nn.Sequential(
            jt.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU_tanh(),
            jt.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def execute(self, img, txt, vec, pe):
        # --- AdaLN 调制 ---
        img_mod1, img_mod2 = self.img_mod(vec)  # 各是 (shift, scale, gate)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # --- 图像流 QKV ---
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1[1]) * img_modulated + img_mod1[0]
        img_qkv = self.img_attn.qkv(img_modulated)
        B, L_img, _ = img_qkv.shape
        img_qkv = img_qkv.reshape(B, L_img, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        img_q, img_k, img_v = img_qkv[0], img_qkv[1], img_qkv[2]
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # --- 文本流 QKV ---
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1[1]) * txt_modulated + txt_mod1[0]
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        B, L_txt, _ = txt_qkv.shape
        txt_qkv = txt_qkv.reshape(B, L_txt, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        txt_q, txt_k, txt_v = txt_qkv[0], txt_qkv[1], txt_qkv[2]
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # --- 拼接后做联合注意力（文本在前，图像在后）---
        q = jt.concat((txt_q, img_q), dim=2)
        k = jt.concat((txt_k, img_k), dim=2)
        v = jt.concat((txt_v, img_v), dim=2)
        attn_out = attention(q, k, v, pe)  # [B, L_txt+L_img, H*D]
        txt_attn_out = attn_out[:, :L_txt, :]
        img_attn_out = attn_out[:, L_txt:, :]

        # --- 图像残差 ---
        img = img + img_mod1[2] * self.img_attn.proj(img_attn_out)
        img = img + img_mod2[2] * self.img_mlp(
            (1 + img_mod2[1]) * self.img_norm2(img) + img_mod2[0]
        )

        # --- 文本残差 ---
        txt = txt + txt_mod1[2] * self.txt_attn.proj(txt_attn_out)
        txt = txt + txt_mod2[2] * self.txt_mlp(
            (1 + txt_mod2[1]) * self.txt_norm2(txt) + txt_mod2[0]
        )

        return img, txt

# ============================================================
# 单流 DiT Block（Flux 末尾几层用）
# ============================================================

class SingleStreamBlock(jt.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.linear1 = jt.nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = jt.nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(self.head_dim)
        self.pre_norm = LayerNormNoAffine(hidden_size, eps=1e-6)
        self.mlp_act = GELU_tanh()
        self.modulation = Modulation(hidden_size, double=False)

    def execute(self, x, vec, pe):
        mod, _ = self.modulation(vec)  # (shift, scale, gate)
        x_mod = (1 + mod[1]) * self.pre_norm(x) + mod[0]

        # 并行线性层：同时产出 QKV 和 MLP 输入
        linear1_out = self.linear1(x_mod)
        qkv = linear1_out[:, :, :self.hidden_size * 3]
        mlp = linear1_out[:, :, self.hidden_size * 3:]

        B, L, _ = qkv.shape
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.norm(q, k, v)

        attn_out = attention(q, k, v, pe)  # [B, L, H*D]

        # 注意力输出 + 激活后的 MLP，拼接后投影
        output = self.linear2(jt.concat((attn_out, self.mlp_act(mlp)), dim=-1))
        x = x + mod[2] * output
        return x

# ============================================================
# 最终输出层
# ============================================================

class LastLayer(jt.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = LayerNormNoAffine(hidden_size, eps=1e-6)
        self.linear = jt.nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = jt.nn.Sequential(
            jt.nn.SiLU(),
            jt.nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def execute(self, x, vec):
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
