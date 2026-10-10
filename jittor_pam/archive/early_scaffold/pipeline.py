import numpy as np
import jittor as jt
from jittor_pam.sampling import (
    get_noise, prepare, get_schedule,
    unpack, denoise_double_control,
)


class DoubleControlSampler:
    """
    Jittor 版双 ControlNet 推理管线。
    对应官方 xflux_pipeline.py 的 DoubleControlSampler。
    文本 embedding（txt, vec）由外部预计算后传入。
    """
    def __init__(self, model, controlnet1, controlnet2, vae):
        self.model = model
        self.controlnet1 = controlnet1
        self.controlnet2 = controlnet2
        self.vae = vae

    def infer(
        self,
        txt, vec,            # 预计算的 T5/CLIP embedding
        depth, normal, hand, seg,  # 条件图 [1, 3, H, W]，范围 [-1, 1]
        height=512, width=512,
        seed=12345,
        num_steps=28,
        guidance=4.0,
        control_gs=(1.0, 1.0, 0, 0),
    ):
        # 1. 生成噪声
        x = get_noise(1, height, width, seed=seed)

        # 2. 准备模型输入（txt/vec 已预计算）
        inp = prepare(x, txt, vec)

        # 3. 采样调度
        image_seq_len = (width // 8) * (height // 8) // (16 * 16)
        timesteps = get_schedule(num_steps, image_seq_len, shift=True)

        # 4. 双 ControlNet 去噪
        x0 = denoise_double_control(
            self.model,
            self.controlnet1,
            self.controlnet2,
            inp,
            controlnet_cond=(depth, normal, hand, seg),
            timesteps=timesteps,
            guidance=guidance,
            control_gs=control_gs,
        )

        # 5. 反 patchify：[B, L, 64] -> [B, 16, H_latent, W_latent]
        x0 = unpack(x0, height, width)

        # 6. VAE 解码
        x0 = self.vae.decode(x0)

        # 7. 后处理：[-1,1] -> [0,255] uint8
        x0 = jt.clamp(x0, -1.0, 1.0)
        x0 = x0[0].permute(1, 2, 0)  # [H, W, 3]
        x0 = (127.5 * (x0 + 1.0)).numpy().astype(np.uint8)

        return x0
