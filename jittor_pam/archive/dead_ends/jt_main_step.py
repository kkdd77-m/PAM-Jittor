import sys, json, numpy as np
sys.path.insert(0, '/data/PAM')
import jittor as jt

# patch: jt.nn.Linear int8 反量化计算（权重 int8 存储 + per-channel scale，fp32 计算）
def _linear_execute_int8(self, x):
    w = self.weight.float32() * self.weight_scale  # [out, in] fp32
    y = jt.matmul(x, w.transpose())
    if self.bias is not None:
        y = y + self.bias
    return y
jt.nn.Linear.execute = _linear_execute_int8

from jittor_pam.model import Flux, FluxParams

def load_safetensors_raw(path):
    """memoryview 零复制读 safetensors，返回 {key: (dtype, shape, raw_bytes)}"""
    with open(path, 'rb') as f:
        n = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        hdr = json.loads(f.read(n))
        data = memoryview(f.read())
    meta = {}
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        start, end = v['data_offsets']
        meta[k] = (v['dtype'], v['shape'], data[start:end])
    return meta

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
jt.flags.use_cuda = 0
model = Flux(params)
print('model instantiated (CPU)', flush=True)
meta = load_safetensors_raw('/data/PAM/weights/flux_dev/flux1-dev.safetensors')
print('tensors:', len(meta), flush=True)

# int8 量化加载（Linear 权重 per-channel int8 + scale；其余 fp16/fp32 直载）
jt.flags.use_cuda = 0  # Var 全 CPU，避免 jt.array 直接占 GPU 导致 OOM
n_q = 0
for key, (d, shape, raw) in meta.items():
    parts = key.split('.')
    mod = get_mod(model, parts[:-1])
    if d == 'BF16':
        u16 = np.frombuffer(raw, dtype='<u2').reshape(shape)
        w = (u16.astype(np.uint32) << 16).view(np.float32)
    elif d == 'F16':
        w = np.frombuffer(raw, dtype='<f2').reshape(shape).astype(np.float32)
    else:
        w = np.frombuffer(raw, dtype='<f4').reshape(shape)
    if parts[-1] == 'weight' and w.ndim == 2:
        absmax = np.abs(w).max(axis=1, keepdims=True)
        absmax = np.where(absmax == 0, 1e-8, absmax)
        scale = absmax / 127.0
        w_q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        mod.weight = jt.array(w_q)
        mod.weight_scale = jt.array(scale.astype(np.float32))
        n_q += 1
    elif parts[-1] == 'weight':
        mod.weight = jt.array(w.astype(np.float16))
    elif parts[-1] == 'bias':
        mod.bias = jt.array(w.astype(np.float32))
    else:
        print('unknown key:', key, flush=True)
    del w
print('quantized Linear:', n_q, flush=True)
del meta  # 释放 24GB 原始文件数据
import gc; gc.collect()
print('img_in dtype:', model.img_in.weight.dtype, flush=True)

jt.flags.use_cuda = 0
jt.flags.lazy_execution = 0  # 急切执行：算完即释放中间张量
inp = {k: jt.array(v) for k, v in np.load('/data/PAM/ref/inp_true.npz').items()}
t_curr = float(np.load('/data/PAM/ref/t_curr.npy'))
t_vec = jt.full((1,), t_curr, dtype=jt.float32)
guidance_vec = jt.full((1,), 4.0, dtype=jt.float32)
print('inputs ready, forward...', flush=True)
pred = model(**inp, timesteps=t_vec, guidance=guidance_vec)
pred = pred.float32()
print('pred shape:', pred.shape, flush=True)
ref = np.load('/data/PAM/ref/main_pred.npy')
p = pred.numpy()
mse = float(((p - ref) ** 2).mean())
mx = float(np.abs(p - ref).max())
print(f'pred MSE {mse:.6e}  MAX {mx:.6e}', flush=True)
print('DONE', flush=True)
