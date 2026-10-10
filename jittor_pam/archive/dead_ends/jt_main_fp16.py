import sys, json, numpy as np, gc
sys.path.insert(0, '/data/PAM')
import jittor as jt

# 关键：use_cuda=1 一开始就开，权重直接在 GPU 分配
jt.flags.use_cuda = 0  # 先 CPU 创建模型和加载权重
jt.flags.lazy_execution = 0

from jittor_pam.model import Flux, FluxParams

def load_safetensors_raw(path):
    with open(path, 'rb') as f:
        n = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        hdr = json.loads(f.read(n))
        data = f.read()
    meta = {}
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        start, end = v['data_offsets']
        meta[k] = (v['dtype'], v['shape'], data[start:end])
    return meta, data

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

print('model instantiated (GPU)', flush=True)
model = Flux(params)

meta, raw_buf = load_safetensors_raw('/data/PAM/weights/flux_dev/flux1-dev.safetensors')
print('tensors:', len(meta), flush=True)

n_loaded = 0
for key, (d, shape, raw) in meta.items():
    parts = key.split('.')
    mod = get_mod(model, parts[:-1])
    if d == 'BF16':
        u16 = np.frombuffer(raw, dtype='<u2').reshape(shape)
        w = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
    elif d == 'F16':
        w = np.frombuffer(raw, dtype='<f2').reshape(shape)
    else:
        w = np.frombuffer(raw, dtype='<f4').reshape(shape).astype(np.float16)
    setattr(mod, parts[-1], jt.array(w))
    n_loaded += 1

del meta, raw_buf
gc.collect()
print('loaded tensors:', n_loaded, flush=True)

# 加载输入
inp = {k: jt.array(v.astype(np.float16)) for k, v in np.load('/data/PAM/ref/inp_true.npz').items()}
t_curr = float(np.load('/data/PAM/ref/t_curr.npy'))
t_vec = jt.full((1,), t_curr, dtype=jt.float16)
guidance_vec = jt.full((1,), 4.0, dtype=jt.float16)

jt.flags.use_cuda = 1  # 加载完切 GPU 跑前向
print('inputs ready, forward...', flush=True)
pred = model(**inp, timesteps=t_vec, guidance=guidance_vec)
pred = pred.float32()

ref = np.load('/data/PAM/ref/main_pred.npy')
a = ref.astype(np.float64)
b = pred.numpy().astype(np.float64)
mse = np.mean((a - b) ** 2)
mx = np.max(np.abs(a - b))
print(f'pred MSE {mse:.6e}  MAX {mx:.6e}', flush=True)
print('DONE', flush=True)
