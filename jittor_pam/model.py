from dataclasses import dataclass
import jittor as jt
from jittor_pam.layers import (
    DoubleStreamBlock, EmbedND, LastLayer,
    MLPEmbedder, SingleStreamBlock, timestep_embedding,
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


class Identity(jt.Module):
    def execute(
        self,
        img,
        img_ids,
        txt,
        txt_ids,
        timesteps,
        y,
        block_controlnet_hidden_states=None,
        guidance=None,
    ):
        import subprocess as _sp
        def _mem():
            return _sp.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.guidance_in is not None:
            if guidance is None:
                raise ValueError("Didn't get guidance strength.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        ids = jt.concat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        controlnet_depth = (
            len(block_controlnet_hidden_states)
            if block_controlnet_hidden_states is not None else 0
        )
        print(f'[mem] after embed: {_mem()} MiB', flush=True)
        for index_block, block in enumerate(self.double_blocks):
            img, txt = block(img, txt, vec, pe)
            if block_controlnet_hidden_states is not None:
                img = img + block_controlnet_hidden_states[index_block % controlnet_depth]
            jt.sync_all()
            print(f'[mem] dbl {index_block}: {_mem()} MiB', flush=True)
        img = jt.concat((txt, img), dim=1)
        for block in self.single_blocks:
            img = block(img, vec, pe)
            jt.sync_all()
        img = img[:, txt.shape[1]:, :]
        return self.final_layer(img, vec)

class Flux(jt.Module):
    """
    Flux DiT 主干：双流 Block → 单流 Block → 最终输出层。
    对应 PyTorch 版 model.py 的 Flux 类（去掉了 gradient checkpointing 和 IP-Adapter）。
    """
    def __init__(self, params: FluxParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = 64

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")

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

        self.double_blocks = jt.nn.ModuleList([
            DoubleStreamBlock(
                self.hidden_size, self.num_heads,
                mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias,
            )
            for _ in range(params.depth)
        ])

        self.single_blocks = jt.nn.ModuleList([
            SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
            for _ in range(params.depth_single_blocks)
        ])

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def execute(
        self,
        img,
        img_ids,
        txt,
        txt_ids,
        timesteps,
        y,
        block_controlnet_hidden_states=None,
        guidance=None,
    ):
        import subprocess as _sp
        def _mem():
            return _sp.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        print(f'[mem] start: {_mem()} MiB', flush=True)

        # --- 输入投影 ---
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.guidance_in is not None:
            if guidance is None:
                raise ValueError("Didn't get guidance strength.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        # --- 位置编码（文本在前，图像在后）---
        ids = jt.concat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        controlnet_depth = (
            len(block_controlnet_hidden_states)
            if block_controlnet_hidden_states is not None else 0
        )

        # --- 双流 Block ---
        for index_block, block in enumerate(self.double_blocks):
            img, txt = block(img, txt, vec, pe)
            if block_controlnet_hidden_states is not None:
                img = img + block_controlnet_hidden_states[index_block % controlnet_depth]

        # --- 单流 Block（文本+图像拼接后一起处理）---
        img = jt.concat((txt, img), dim=1)
        for block in self.single_blocks:
            img = block(img, vec, pe)
            jt.gc()  # 回收内存池

        # --- 截掉文本 token，只保留图像部分 ---
        img = img[:, txt.shape[1]:, :]

        # --- 最终输出 ---
        img = self.final_layer(img, vec)
        return img
