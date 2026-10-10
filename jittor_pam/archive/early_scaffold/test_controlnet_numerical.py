import sys
import numpy as np
import torch
import jittor as jt

jt.flags.use_cuda = 1

sys.path.insert(0, '/data/PAM/appearance_gen/src')
from flux.controlnet import ControlNetFlux as PTControlNet, FluxParams as PTParams

sys.path.insert(0, '/data/PAM')
from jittor_pam.controlnet import ControlNetFlux as JTControlNet, FluxParams as JTParams

params_kwargs = dict(
    in_channels=64, vec_in_dim=256, context_in_dim=256,
    hidden_size=64, mlp_ratio=4.0, num_heads=4,
    depth=2, depth_single_blocks=2,
    axes_dim=[8, 8], theta=10000, qkv_bias=False, guidance_embed=False,
)
controlnet_depth = 2
condition_in_channels = 3

pt_model = PTControlNet(PTParams(**params_kwargs), controlnet_depth, condition_in_channels).cuda().eval()
jt_model = JTControlNet(JTParams(**params_kwargs), controlnet_depth, condition_in_channels)

# 权重拷贝
pt_state = pt_model.state_dict()
jt_state = {k: v.cpu().numpy() for k, v in pt_state.items()}
jt_keys = set(jt_model.state_dict().keys())
pt_keys = set(pt_state.keys())
if jt_keys != pt_keys:
    print("KEY MISMATCH!")
    print(f"PT-only: {sorted(pt_keys - jt_keys)[:15]}")
    print(f"JT-only: {sorted(jt_keys - pt_keys)[:15]}")
    print(f"PT total: {len(pt_keys)}, JT total: {len(jt_keys)}")
    sys.exit(1)
print(f"All {len(pt_keys)} keys matched.")
jt_model.load_state_dict(jt_state)

# 构造输入
np.random.seed(42)
B = 1
L_img = 16  # 4x4 latent spatial
L_txt = 8

img = np.random.randn(B, L_img, 64).astype(np.float32)
controlnet_cond = np.random.randn(B, 3, 64, 64).astype(np.float32)  # 条件图
txt = np.random.randn(B, L_txt, 256).astype(np.float32)
timesteps = np.array([0.5], dtype=np.float32)
y = np.random.randn(B, 256).astype(np.float32)

h = w = 4
img_ids = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), -1).reshape(1, -1, 2).astype(np.float32)
img_ids = np.broadcast_to(img_ids, (B, L_img, 2)).copy()
txt_ids = np.zeros((B, L_txt, 2), dtype=np.float32)

# PyTorch
with torch.no_grad():
    pt_outs = pt_model(
        img=torch.tensor(img).cuda(),
        img_ids=torch.tensor(img_ids).cuda(),
        controlnet_cond=torch.tensor(controlnet_cond).cuda(),
        txt=torch.tensor(txt).cuda(),
        txt_ids=torch.tensor(txt_ids).cuda(),
        timesteps=torch.tensor(timesteps).cuda(),
        y=torch.tensor(y).cuda(),
    )

# Jittor
jt_outs = jt_model(
    img=jt.array(img),
    img_ids=jt.array(img_ids),
    controlnet_cond=jt.array(controlnet_cond),
    txt=jt.array(txt),
    txt_ids=jt.array(txt_ids),
    timesteps=jt.array(timesteps),
    y=jt.array(y),
)

print(f"\nOutput: {len(pt_outs)} residuals")
all_pass = True
for i, (pt_o, jt_o) in enumerate(zip(pt_outs, jt_outs)):
    pt_np = pt_o.detach().cpu().numpy()
    jt_np = jt_o.numpy()
    mse = np.mean((pt_np - jt_np) ** 2)
    maxdiff = np.max(np.abs(pt_np - jt_np))
    ok = mse < 1e-3
    print(f"  residual[{i}]: shape={pt_np.shape} MSE={mse:.2e} MaxDiff={maxdiff:.2e} [{'PASS' if ok else 'FAIL'}]")
    all_pass &= ok

print("\n*** CONTROLNET PASSED ***" if all_pass else "\n*** FAILED ***")
