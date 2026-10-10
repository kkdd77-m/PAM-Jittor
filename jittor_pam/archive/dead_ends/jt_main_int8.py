import sys, json, numpy as np, gc
sys.path.insert(0, '/data/PAM')
import jittor as jt
jt.flags.use_cuda = 0
jt.flags.lazy_execution = 0

def _linear_execute_int8(self, x):
    w = self.weight.float32() * self.weight_scale
    y = jt.matmul(x, w.transpose())
    if self.bias is not None:
        y = y + self.bias.float32()
    return y
jt.nn.Linear.execute = _linear_execute_int8

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
print('loading int8 weights...', flush=True)

import mmap
path = '/data/PAM/weights/flux_dev/flux1-dev.safetensors'
f = open(path, 'rb')
n = np.frombuffer(f.read(8), dtype=np.uint64)[0]
hdr = json.loads(f.read(n))
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_start = 8 + n

n_q = 0
for k, v in hdr.items():
    if k == '__metadata__':
        continue
    parts = k.split('.')
    mod = get_mod(model, parts[:-1])
    start = int(data_start + v['data_offsets'][0])
    end = int(data_start + v['data_offsets'][1])
    shape = v['shape']
    d = v['dtype']
    if d == 'BF16':
        u16 = np.frombuffer(bytes(mm[start:end]), dtype='<u2').reshape(shape)
        w = (u16.astype(np.uint32) << 16).view(np.float32)
    elif d == 'F16':
        w = np.frombuffer(bytes(mm[start:end]), dtype='<f2').reshape(shape).astype(np.float32)
    else:
        w = np.frombuffer(bytes(mm[start:end]), dtype='<f4').reshape(shape)
    if parts[-1] == 'weight' and w.ndim == 2:
        absmax = np.abs(w).max(axis=1, keepdims=True)
        absmax = np.where(absmax == 0, 1e-8, absmax)
        scale = absmax / 127.0
        w_q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        mod.weight = jt.array(w_q)
        mod.weight_scale = jt.array(scale.astype(np.float32))
        n_q += 1
    elif parts[-1] == 'weight':
        mod.weight = jt.array(w.astype(np.float32))
    elif parts[-1] == 'bias':
        mod.bias = jt.array(w.astype(np.float32))
    del w

mm.close()
f.close()
gc.collect()
print(f'quantized Linear: {n_q}, switching to GPU...', flush=True)

# jt.flags.use_cuda = 1  # 留 CPU 跑

inp = {k: jt.array(v.astype(np.float32)) for k, v in np.load('/data/PAM/ref/inp_true.npz').items()}
t_curr = float(np.load('/data/PAM/ref/t_curr.npy'))
t_vec = jt.full((1,), t_curr, dtype=jt.float32)
guidance_vec = jt.full((1,), 4.0, dtype=jt.float32)

print('forward...', flush=True)
pred = model(**inp, timesteps=t_vec, guidance=guidance_vec)

ref = np.load('/data/PAM/ref/main_pred.npy')
a = ref.astype(np.float64)
b = pred.numpy().astype(np.float64)
mse = np.mean((a - b) ** 2)
mx = np.max(np.abs(a - b))
print(f'pred MSE {mse:.6e}  MAX {mx:.6e}', flush=True)
print('DONE', flush=True)
