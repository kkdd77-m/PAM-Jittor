import sys, json, numpy as np, gc
sys.path.insert(0, '/data/PAM')
import jittor as jt
jt.flags.use_cuda = 0
jt.flags.lazy_execution = 0

# patch Linear: 统一 cast 到 weight dtype，避免 mixed dtype
def _linear_execute(self, x):
    w = self.weight
    if x.dtype != w.dtype:
        x = x.cast(w.dtype)
    y = jt.matmul(x, w.transpose())
    if self.bias is not None:
        b = self.bias
        if b.dtype != y.dtype:
            b = b.cast(y.dtype)
        y = y + b
    return y
jt.nn.Linear.execute = _linear_execute

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

print('model creating (CPU)...', flush=True)
model = Flux(params)
gc.collect()
print('loading weights (fp16)...', flush=True)

# 用 torch venv 已经转好的方式读：直接用 safetensors 库（torch venv 有，但 jittor python 可能没有）
# 用手动读取，不 mmap，逐 tensor 读
path = '/data/PAM/weights/flux_dev/flux1-dev.safetensors'
import mmap
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
        w = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
    elif d == 'F16':
        w = np.frombuffer(bytes(mm[start:end]), dtype='<f2').reshape(shape).copy()
    else:
        w = np.frombuffer(bytes(mm[start:end]), dtype='<f4').reshape(shape).astype(np.float16)
    setattr(mod, parts[-1], jt.array(w))
    del w

mm.close()
f.close()
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
