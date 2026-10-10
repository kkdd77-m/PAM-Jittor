import math
import numpy as np
import jittor as jt


def get_noise(num_samples, height, width, seed=None):
    """生成 latent 空间噪声。形状 [B, 16, 2*ceil(H/16), 2*ceil(W/16)]"""
    rng = np.random.RandomState(seed)
    h = 2 * math.ceil(height / 16)
    w = 2 * math.ceil(width / 16)
    noise = rng.randn(num_samples, 16, h, w).astype(np.float32)
    return jt.array(noise)


def prepare(img, txt, vec):
    """
    准备模型输入。txt 和 vec 是预计算的 T5/CLIP embedding。
    img: [B, C, H, W] latent
    txt: [B, L_txt, D_txt] T5 embedding
    vec: [B, D_vec] CLIP pooled embedding
    返回 dict，含 img/img_ids/txt/txt_ids/y
    """
    bs, c, h, w = img.shape

    # patchify: [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
    img = img.reshape(bs, c, h // 2, 2, w // 2, 2)
    img = img.permute(0, 2, 4, 1, 3, 5)
    img = img.reshape(bs, (h // 2) * (w // 2), c * 4)

    # 3D position ids: [B, L, 3]，第0维恒0，第1维行号，第2维列号
    img_ids = np.zeros((h // 2, w // 2, 3), dtype=np.float32)
    img_ids[..., 1] = np.arange(h // 2)[:, None]
    img_ids[..., 2] = np.arange(w // 2)[None, :]
    img_ids = np.broadcast_to(img_ids, (bs, h // 2, w // 2, 3)).copy()
    img_ids = img_ids.reshape(bs, -1, 3)
    img_ids = jt.array(img_ids)

    # txt_ids: 全零 [B, L_txt, 3]
    txt_ids = jt.zeros((bs, txt.shape[1], 3))

    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "y": vec,
    }


def time_shift(mu, sigma, t):
    """时间偏移函数（对应 sampling.py 的 time_shift）"""
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def get_lin_function(x1=256, y1=0.5, x2=4096, y2=1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15, shift=True):
    """生成采样时间步调度"""
    timesteps = np.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def unpack(x, height, width):
    """反 patchify：[B, L, C*4] -> [B, C, H, W]"""
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)
    bs = x.shape[0]
    c = x.shape[-1] // 4  # ph*pw=4
    x = x.reshape(bs, h, w, c, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)  # [B, C, h, 2, w, 2]
    x = x.reshape(bs, c, h * 2, w * 2)
    return x


def denoise_double_control(
    model, controlnet1, controlnet2,
    inp, controlnet_cond,
    timesteps, guidance=4.0, control_gs=(1.0, 1.0, 0, 0),
):
    """
    PAM 双 ControlNet 采样（对应 denoise_double_control）。
    controlnet_cond = (depth, normal, hand, seg)，每个 [B, 3, H, W]。
    """
    img = inp["img"]
    guidance_vec = jt.full((img.shape[0],), guidance, dtype=img.dtype)
    depth, normal, hand, seg = controlnet_cond

    # 去掉多余的 batch 维
    if len(depth.shape) > 4:
        depth = depth.squeeze(0)
    if len(normal.shape) > 4:
        normal = normal.squeeze(0)
    if len(hand.shape) > 4:
        hand = hand.squeeze(0)
    if len(seg.shape) > 4:
        seg = seg.squeeze(0)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = jt.full((img.shape[0],), t_curr, dtype=img.dtype)

        physical_latent = jt.concat([depth, normal], dim=1)  # 6ch
        semantic_latent = jt.concat([hand, seg], dim=1)      # 6ch

        inp["img"] = img

        # ControlNet1: 物理条件
        block_res_1 = controlnet1(
            **inp,
            controlnet_cond=physical_latent,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        # ControlNet2: 语义条件
        block_res_2 = controlnet2(
            **inp,
            controlnet_cond=semantic_latent,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        # 残差加权相加，注入主模型
        pred = model(
            **inp,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[
                i * control_gs[0] + j * control_gs[1]
                for i, j in zip(block_res_1, block_res_2)
            ],
        )

        # Euler 步
        img = img + (t_prev - t_curr) * pred

    return img
