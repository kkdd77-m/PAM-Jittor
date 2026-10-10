import sys
import numpy as np
import jittor as jt

jt.flags.use_cuda = 1

sys.path.insert(0, '/data/PAM')
from jittor_pam.model import Flux, FluxParams
from jittor_pam.controlnet import ControlNetFlux
from jittor_pam.autoencoder import AutoEncoder  # 确认类名
from jittor_pam.pipeline import DoubleControlSampler

# 小模型参数（验证管线连通性，不验证生成质量）
params = FluxParams(
    in_channels=64, vec_in_dim=768, context_in_dim=4096,
    hidden_size=64, mlp_ratio=4.0, num_heads=4,
    depth=2, depth_single_blocks=2,
    axes_dim=[4, 8, 4], theta=10000, qkv_bias=False, guidance_embed=True,
)

print("Building models...")
model = Flux(params)
cn1 = ControlNetFlux(params, controlnet_depth=2, condition_in_channels=6)
cn2 = ControlNetFlux(params, controlnet_depth=2, condition_in_channels=6)

from jittor_pam.autoencoder import AutoEncoder, AutoEncoderParams
vae_params = AutoEncoderParams(
    resolution=256, in_channels=3, ch=32, out_ch=3,
    ch_mult=[1, 2, 4, 4], num_res_blocks=2, z_channels=16,
    scale_factor=0.3611, shift_factor=0.0,
)
vae = AutoEncoder(vae_params)
print("VAE initialized (ch=32 for smoke test)")

sampler = DoubleControlSampler(model, cn1, cn2, vae)

# 构造假输入
H, W = 256, 256
txt = jt.array(np.random.randn(1, 512, 4096).astype(np.float32))
vec = jt.array(np.random.randn(1, 768).astype(np.float32))
depth = jt.array(np.random.uniform(-1, 1, (1, 3, H, W)).astype(np.float32))
normal = jt.array(np.random.uniform(-1, 1, (1, 3, H, W)).astype(np.float32))
hand = jt.array(np.random.uniform(-1, 1, (1, 3, H, W)).astype(np.float32))
seg = jt.array(np.random.uniform(-1, 1, (1, 3, H, W)).astype(np.float32))

print(f"\nRunning inference ({H}x{W}, 28 steps)...")
result = sampler.infer(
    txt=txt, vec=vec,
    depth=depth, normal=normal, hand=hand, seg=seg,
    height=H, width=W, seed=42, num_steps=28,
)

print(f"\nOutput shape: {result.shape} (expect [{H}, {W}, 3])")
print(f"Output range: [{result.min()}, {result.max()}] (expect [0, 255])")
print(f"dtype: {result.dtype}")

ok = result.shape == (H, W, 3) and result.dtype == np.uint8
print("\n*** PIPELINE SMOKE TEST PASSED ***" if ok else "\n*** FAILED ***")
