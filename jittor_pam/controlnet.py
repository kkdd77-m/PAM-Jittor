from dataclasses import dataclass
import jittor as jt
from jittor_pam.layers import (
    DoubleStreamBlock, EmbedND,
    MLPEmbedder, timestep_embedding,
)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool


def zero_module(module):
    """将模块参数全部置零（ControlNet 标准初始化）"""
    sd = module.state_dict()
    for k in sd:
        sd[k] = jt.zeros_like(sd[k])
    module.load_state_dict(sd)
    return module


class SiLU(jt.Module):
    def execute(self, x):
        return x * jt.sigmoid(x)


class ControlNetFlux(jt.Module):
    """
    Flux ControlNet：输入条件图，输出与 double_blocks 对应的残差。
    对应 PyTorch 版 controlnet.py 的 ControlNetFlux 类。
    """
    def __init__(self, params: FluxParams, controlnet_depth=2, condition_in_channels=3):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels

        pe_dim = params.hidden_size // params.num_heads
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = jt.nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if params.guidance_embed else None
        )
        self.txt_in = jt.nn.Linear(params.context_in_dim, self.hidden_size)

        # ControlNet 专用的双流 Block（数量少于主模型）
        self.double_blocks = jt.nn.ModuleList([
            DoubleStreamBlock(
                self.hidden_size, self.num_heads,
                mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias,
            )
            for _ in range(controlnet_depth)
        ])

        # 零初始化的残差输出层
        self.controlnet_blocks = jt.nn.ModuleList([
            zero_module(jt.nn.Linear(self.hidden_size, self.hidden_size))
            for _ in range(controlnet_depth)
        ])

        # 条件投影（patchify 后从 64 维 → hidden_size）
        self.pos_embed_input = jt.nn.Linear(self.in_channels, self.hidden_size, bias=True)

        # 条件图 CNN 处理（3 次 stride=2，共 8x 下采样）
        self.input_hint_block = jt.nn.Sequential(
            jt.nn.Conv(condition_in_channels, 16, 3, padding=1),
            SiLU(),
            jt.nn.Conv(16, 16, 3, padding=1),
            SiLU(),
            jt.nn.Conv(16, 16, 3, padding=1, stride=2),
            SiLU(),
            jt.nn.Conv(16, 16, 3, padding=1),
            SiLU(),
            jt.nn.Conv(16, 16, 3, padding=1, stride=2),
            SiLU(),
            jt.nn.Conv(16, 16, 3, padding=1),
            SiLU(),
            jt.nn.Conv(16, 16, 3, padding=1, stride=2),
            SiLU(),
            zero_module(jt.nn.Conv(16, 16, 3, padding=1)),
        )

    def execute(
        self,
        img,
        img_ids,
        controlnet_cond,
        txt,
        txt_ids,
        timesteps,
        y,
        guidance=None,
    ):
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # --- 图像投影 ---
        img = self.img_in(img)

        # --- 条件图处理：CNN → patchify → 投影 → 加到 img 上 ---
        controlnet_cond = self.input_hint_block(controlnet_cond)
        # rearrange "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
        B, C, H, W = controlnet_cond.shape
        ph, pw = 2, 2
        controlnet_cond = controlnet_cond.reshape(B, C, H // ph, ph, W // pw, pw)
        controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
        controlnet_cond = controlnet_cond.reshape(B, (H // ph) * (W // pw), C * ph * pw)
        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond

        # --- 时间/向量/文本投影 ---
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.guidance_in is not None:
            if guidance is None:
                raise ValueError("Didn't get guidance strength.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        # --- 位置编码 ---
        ids = jt.concat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # --- 双流 Block，收集每层输出 ---
        block_res_samples = []
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, pe)
            block_res_samples.append(img)

        # --- 零初始化 Linear 产生残差 ---
        controlnet_block_res_samples = []
        for res_sample, cn_block in zip(block_res_samples, self.controlnet_blocks):
            controlnet_block_res_samples.append(cn_block(res_sample))

        return tuple(controlnet_block_res_samples)
