import sys, json, numpy as np, gc
sys.path.insert(0, '/data/PAM')
import jittor as jt
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 0

# tiny init：模型创建不分配随机权重
_orig_init = jt.nn.Linear.__init__
def _tiny_init(self, in_features, out_features, bias=True):
    self.in_features = in_features
    self.out_features = out_features
    self.weight = jt.zeros((1,1), dtype=jt.float32)
    self.bias = jt.zeros((1,), dtype=jt.float32) if bias else None
jt.nn.Linear.__init__ = _tiny_init

# int8 反量化
def _lin_int8(self, x):
    w = self.weight.float32() * self.weight_scale
    y = jt.matmul(x, w.transpose())
    if self.bias is not None:
        y = y + self.bias.float32()
    return y
jt.nn.Linear.execute = _lin_int8

from jittor_pam.model import Flux, FluxParams
def get_mod(m, parts):
    for p in parts:
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m

params = FluxParams(
    in_channels=64, vec_in_dim=768, context_in_dim=4096,
    hidden_size=3072, mlp_ratio=4.0, num_heads=24,
    depth=19, depth_single_blocks=38,
    axes_dim=[16, 56, 56], theta=10000,
    qkv_bias=True, guidance_embed=True,
)
print('creating model...', flush=True)
model = Flux(params)
print('loading int8 weights directly to GPU...', flush=True)

path = '/data/PAM/weights/flux_dev/flux1-dev.safetensors'
f = open(path, 'rb')
n = np.frombuffer(f.read(8), dtype=np.uint64)[0]
hdr = json.loads(f.read(n))
ds = 8 + n
n_q = 0
for k, v in hdr.items():
    if k == '__metadata__': continue
    parts = k.split('.')
    mod = get_mod(model, parts[:-1])
    s0 = int(ds + v['data_offsets'][0])
    s1 = int(ds + v['data_offsets'][1])
    shape, d = v['shape'], v['dtype']
    f.seek(s0); raw = f.read(s1 - s0)
    if d == 'BF16':
        u16 = np.frombuffer(raw, dtype='<u2').reshape(shape)
        w = (u16.astype(np.uint32) << 16).view(np.float32)
    elif d == 'F16':
        w = np.frombuffer(raw, dtype='<f2').reshape(shape).astype(np.float32)
    else:
        w = np.frombuffer(raw, dtype='<f4').reshape(shape)
    if parts[-1] == 'weight' and w.ndim == 2:
        ab = np.abs(w).max(axis=1, keepdims=True)
        ab = np.where(ab == 0, 1e-8, ab)
        sc = ab / 127.0
        wq = np.clip(np.round(w / sc), -127, 127).astype(np.int8)
        mod.weight = jt.array(wq)
        mod.weight_scale = jt.array(sc.astype(np.float32))
        n_q += 1
    elif parts[-1] == 'weight':
        mod.weight = jt.array(w.astype(np.float32))
    elif parts[-1] == 'bias':
        mod.bias = jt.array(w.astype(np.float32))
    del w, raw
f.close()
gc.collect()
print(f'quantized: {n_q}, forward...', flush=True)

inp = {k: jt.array(v.astype(np.float32)) for k, v in np.load('/data/PAM/ref/inp_true.npz').items()}
t_vec = jt.full((1,), float(np.load('/data/PAM/ref/t_curr.npy')), dtype=jt.float32)
g_vec = jt.full((1,), 4.0, dtype=jt.float32)

pred = model(**inp, timesteps=t_vec, guidance=g_vec)

ref = np.load('/data/PAM/ref/main_pred.npy')
a = ref.astype(np.float64); b = pred.numpy().astype(np.float64)
print(f'pred MSE {np.mean((a-b)**2):.6e}  MAX {np.max(np.abs(a-b)):.6e}', flush=True)
print('DONE', flush=True)
