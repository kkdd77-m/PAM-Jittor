import sys, json, numpy as np, gc
sys.path.insert(0, '/data/PAM')
import jittor as jt
from safetensors import safe_open

jt.flags.use_cuda = 0
jt.flags.lazy_execution = 0

from jittor_pam.model import Flux, FluxParams

def get_mod(m, parts):
    for p in parts:
        if p.isdigit():
            m = m[int(p)]
        else:
            m = getattr(m, p)
    return m

params = FluxParams(
    in_channels=64, vec_in_dim=768, context_in_dim=4096,
    hidden_size=3072, mlp_ratio=4.0, num_heads=24,
    depth=19, depth_single_blocks=38,
    axes_dim=[16, 56, 56], theta=10000,
    qkv_bias=True, guidance_embed=True,
)

print('model creating...', flush=True)
model = Flux(params)
gc.collect()
print('model created, loading weights...', flush=True)

# 用 mmap 逐 tensor 读，不一次性载入
with safe_open('/data/PAM/weights/flux_dev/flux1-dev.safetensors', framework='np', device='cpu') as f:
    for key in f.keys():
        raw = f.get_tensor(key)  # numpy 数组（mmap 视图）
        parts = key.split('.')
        mod = get_mod(model, parts[:-1])
        # 统一转 fp16
        w = raw.astype(np.float16) if raw.dtype != np.float16 else raw
        setattr(mod, parts[-1], jt.array(w))
        del raw, w

gc.collect()
print('weights loaded, switching to GPU...', flush=True)

jt.flags.use_cuda = 1

inp = {k: jt.array(v.astype(np.float16)) for k, v in np.load('/data/PAM/ref/inp_true.npz').items()}
t_curr = float(np.load('/data/PAM/ref/t_curr.npy'))
t_vec = jt.full((1,), t_curr, dtype=jt.float16)
guidance_vec = jt.full((1,), 4.0, dtype=jt.float16)

print('forward...', flush=True)
pred = model(**inp, timesteps=t_vec, guidance=guidance_vec)
pred = pred.float32()

ref = np.load('/data/PAM/ref/main_pred.npy')
a = ref.astype(np.float64)
b = pred.numpy().astype(np.float64)
mse = np.mean((a - b) ** 2)
mx = np.max(np.abs(a - b))
print(f'pred MSE {mse:.6e}  MAX {mx:.6e}', flush=True)
print('DONE', flush=True)
