import sys, json, numpy as np, gc, mmap
sys.path.insert(0, '/data/PAM')
import jittor as jt

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
print('model created, loading weights (fp32, mmap)...', flush=True)

path = '/data/PAM/weights/flux_dev/flux1-dev.safetensors'
f = open(path, 'rb')
n = np.frombuffer(f.read(8), dtype=np.uint64)[0]
hdr = json.loads(f.read(n))
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
data_start = 8 + n

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
        w = (u16.astype(np.uint32) << 16).view(np.float32).copy()
    elif d == 'F16':
        w = np.frombuffer(bytes(mm[start:end]), dtype='<f2').reshape(shape).astype(np.float32)
    else:
        w = np.frombuffer(bytes(mm[start:end]), dtype='<f4').reshape(shape).copy()
    setattr(mod, parts[-1], jt.array(w))
    del w

mm.close()
f.close()
gc.collect()
print('weights loaded, forward on CPU...', flush=True)

inp = {k: jt.array(v.astype(np.float32)) for k, v in np.load('/data/PAM/ref/inp_true.npz').items()}
t_curr = float(np.load('/data/PAM/ref/t_curr.npy'))
t_vec = jt.full((1,), t_curr, dtype=jt.float32)
guidance_vec = jt.full((1,), 4.0, dtype=jt.float32)

pred = model(**inp, timesteps=t_vec, guidance=guidance_vec)
pred = pred.float32()

ref = np.load('/data/PAM/ref/main_pred.npy')
a = ref.astype(np.float64)
b = pred.numpy().astype(np.float64)
mse = np.mean((a - b) ** 2)
mx = np.max(np.abs(a - b))
print(f'pred MSE {mse:.6e}  MAX {mx:.6e}', flush=True)
print('DONE', flush=True)
