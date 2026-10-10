import sys, json, numpy as np, gc, subprocess
sys.path.insert(0, '/data/PAM')
import jittor as jt
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 0
jt.flags.device_mem_limit = 30064771072  # 28GB

def mem():
    return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()

_oi = jt.nn.Linear.__init__
def _ti(self, i, o, bias=True):
    self.in_features=i; self.out_features=o
    self.weight=jt.zeros((1,1),dtype=jt.float32)
    self.bias=jt.zeros((1,),dtype=jt.float32) if bias else None
jt.nn.Linear.__init__=_ti

# fp16 Linear：统一 cast
def _li(self,x):
    w=self.weight
    if x.dtype!=w.dtype: x=x.cast(w.dtype)
    y=jt.matmul(x,w.transpose())
    if self.bias is not None:
        b=self.bias
        if b.dtype!=y.dtype: b=b.cast(y.dtype)
        y=y+b
    return y
jt.nn.Linear.execute=_li

from jittor_pam.model import Flux, FluxParams
from jittor_pam.layers import timestep_embedding

def get_mod(m, parts):
    for p in parts:
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m

params = FluxParams(
    in_channels=64, vec_in_dim=768, context_in_dim=4096,
    hidden_size=3072, mlp_ratio=4.0, num_heads=24,
    depth=19, depth_single_blocks=38,
    axes_dim=[16,56,56], theta=10000, qkv_bias=True, guidance_embed=True)

print('creating model...', flush=True)
model = Flux(params)
print('loading fp16 to GPU...', flush=True)

path='/data/PAM/weights/flux_dev/flux1-dev.safetensors'
f=open(path,'rb')
n=np.frombuffer(f.read(8),dtype=np.uint64)[0]
hdr=json.loads(f.read(n)); ds=8+n
for k,v in hdr.items():
    if k=='__metadata__': continue
    parts=k.split('.'); mod=get_mod(model,parts[:-1])
    s0=int(ds+v['data_offsets'][0]); s1=int(ds+v['data_offsets'][1])
    shape,d=v['shape'],v['dtype']
    f.seek(s0); raw=f.read(s1-s0)
    if d=='BF16':
        u16=np.frombuffer(raw,dtype='<u2').reshape(shape)
        w=(u16.astype(np.uint32)<<16).view(np.float32).astype(np.float16)
    elif d=='F16':
        w=np.frombuffer(raw,dtype='<f2').reshape(shape).copy()
    else:
        w=np.frombuffer(raw,dtype='<f4').reshape(shape).astype(np.float16)
    setattr(mod, parts[-1], jt.array(w))
    del w,raw
f.close()
jt.clean_graph(); jt.gc(); gc.collect()
print(f'loaded, mem: {mem()} MiB', flush=True)

inp = {k: jt.array(v.astype(np.float16)) for k,v in np.load('/data/PAM/ref/inp_true.npz').items()}
t_vec = jt.full((1,), float(np.load('/data/PAM/ref/t_curr.npy')), dtype=jt.float16)
g_vec = jt.full((1,), 4.0, dtype=jt.float16)
img=inp['img']; img_ids=inp['img_ids']; txt=inp['txt']; txt_ids=inp['txt_ids']; y=inp['y']

print('projections...', flush=True)
img=model.img_in(img)
vec=model.time_in(timestep_embedding(t_vec,256))
vec=vec+model.guidance_in(timestep_embedding(g_vec,256))
vec=vec+model.vector_in(y)
txt=model.txt_in(txt)
ids=jt.concat((txt_ids,img_ids),dim=1)
pe=model.pe_embedder(ids)
jt.clean_graph(); jt.gc()
print(f'after proj: {mem()} MiB', flush=True)

for i,block in enumerate(model.double_blocks):
    img,txt=block(img,txt,vec,pe)
    jt.clean_graph(); jt.gc()
    if i%5==0: print(f'  dbl {i:2d}: {mem()} MiB', flush=True)

img=jt.concat((txt,img),dim=1)
jt.clean_graph(); jt.gc()

for i,block in enumerate(model.single_blocks):
    img=block(img,vec,pe)
    jt.clean_graph(); jt.gc()
    if i%5==0: print(f'  sgl {i:2d}: {mem()} MiB', flush=True)

img=img[:,txt.shape[1]:,:]
img=model.final_layer(img,vec)
pred=img.float32()
print(f'done: {mem()} MiB', flush=True)

ref=np.load('/data/PAM/ref/main_pred.npy')
a=ref.astype(np.float64); b=pred.numpy().astype(np.float64)
print(f'pred MSE {np.mean((a-b)**2):.6e}  MAX {np.max(np.abs(a-b)):.6e}', flush=True)
print('DONE', flush=True)
