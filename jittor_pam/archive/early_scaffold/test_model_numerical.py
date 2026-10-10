import sys
import numpy as np
import torch
import jittor as jt

jt.flags.use_cuda = 1

# PyTorch
sys.path.insert(0, '/data/PAM/appearance_gen/src')
from flux.model import Flux as PTFlux, FluxParams as PTFluxParams

# Jittor
sys.path.insert(0, '/data/PAM')
from jittor_pam.model import Flux as JTFlux, FluxParams as JTFluxParams

# --- 用小尺寸快速验证 ---
params_kwargs = dict(
    in_channels=16,
    vec_in_dim=256,
    context_in_dim=256,
    hidden_size=64,
    mlp_ratio=4.0,
    num_heads=4,
    depth=2,
    depth_single_blocks=2,
    axes_dim=[8, 8],
    theta=10000,
    qkv_bias=False,
    guidance_embed=False,
)

pt_params = PTFluxParams(**params_kwargs)
jt_params = JTFluxParams(**params_kwargs)

pt_model = PTFlux(pt_params).cuda().eval()
jt_model = JTFlux(jt_params)

# --- 权重拷贝 ---
pt_state = pt_model.state_dict()
jt_state = {k: v.cpu().numpy() for k, v in pt_state.items()}
jt_keys = set(jt_model.state_dict().keys())
pt_keys = set(pt_state.keys())
if jt_keys != pt_keys:
    print("KEY MISMATCH!")
    print(f"PT-only: {sorted(pt_keys - jt_keys)[:10]}")
    print(f"JT-only: {sorted(jt_keys - pt_keys)[:10]}")
    print(f"PT total: {len(pt_keys)}, JT total: {len(jt_keys)}")
    sys.exit(1)
print(f"All {len(pt_keys)} keys matched.")
jt_model.load_state_dict(jt_state)

# --- 构造输入 ---
np.random.seed(42)
B = 1
L_img = 16   # 4x4 spatial
L_txt = 8

img = np.random.randn(B, L_img, 16).astype(np.float32)
txt = np.random.randn(B, L_txt, 256).astype(np.float32)
timesteps = np.array([0.5], dtype=np.float32)
y = np.random.randn(B, 256).astype(np.float32)

# img_ids: 2D 网格坐标 [B, L_img, 2]
h = w = 4
img_ids = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), -1).reshape(1, -1, 2).astype(np.float32)
img_ids = np.broadcast_to(img_ids, (B, L_img, 2)).copy()
# txt_ids: 全零（文本无空间位置）
txt_ids = np.zeros((B, L_txt, 2), dtype=np.float32)

# --- PyTorch 前向 ---
with torch.no_grad():
    pt_out = pt_model(
        img=torch.tensor(img).cuda(),
        img_ids=torch.tensor(img_ids).cuda(),
        txt=torch.tensor(txt).cuda(),
        txt_ids=torch.tensor(txt_ids).cuda(),
        timesteps=torch.tensor(timesteps).cuda(),
        y=torch.tensor(y).cuda(),
    )

# --- Jittor 前向 ---
jt_out = jt_model(
    img=jt.array(img),
    img_ids=jt.array(img_ids),
    txt=jt.array(txt),
    txt_ids=jt.array(txt_ids),
    timesteps=jt.array(timesteps),
    y=jt.array(y),
)

pt_np = pt_out.detach().cpu().numpy()
jt_np = jt_out.numpy()

mse = np.mean((pt_np - jt_np) ** 2)
maxdiff = np.max(np.abs(pt_np - jt_np))

print(f"\nOutput shape: PT={pt_np.shape}, JT={jt_np.shape}")
print(f"MSE={mse:.2e}  MaxDiff={maxdiff:.2e}")
if mse < 1e-3:
    print("*** FLUX MODEL NUMERICAL ALIGNMENT PASSED ***")
else:
    print("*** FAILED ***")
