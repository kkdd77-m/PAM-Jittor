import sys, json, numpy as np, gc, subprocess
sys.path.insert(0, '/data/PAM')
import jittor as jt
jt.flags.use_cuda = 1
jt.flags.lazy_execution = 0
jt.flags.device_mem_limit = 33285996544  # 31GB

def mem():
    return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()

# tiny init
_oi = jt.nn.Linear.__init__
def _ti(self, i, o, bias=True):
    self.in_features=i; self.out_features=o
    self.weight=jt.zeros((1,1),dtype=jt.float32)
    self.bias=jt.zeros((1,),dtype=jt.float32) if bias else None
jt.nn.Linear.__init__=_ti

# fp16 存储 / fp32 计算
def _li(self,x):
    w=self.weight.float32()
    if x.dtype!=jt.float32: x=x.float32()
    y=jt.matmul(x,w.transpose())
    if self.bias is not None: y=y+self.bias.float32()
    del w
    return y
jt.nn.Linear.execute=_li

from jittor_pam.model import Flux, FluxParams
from jittor_pam.controlnet import ControlNetFlux
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

# ---- 主模型 fp16 权重 ----
print('creating main model...', flush=True)
model = Flux(params)
print('loading fp16 main weights...', flush=True)
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
f.close(); jt.clean_graph(); jt.gc(); gc.collect()
print(f'main loaded: {mem()} MiB', flush=True)

# ---- 双 ControlNet fp32：恢复正常 Linear 构造（CN 很小）----
jt.nn.Linear.__init__ = _oi
print('loading controlnets...', flush=True)
sd = np.load('/data/PAM/ref/controlnet_fp16.npz')
def load_cn(prefix):
    cn = ControlNetFlux(params, controlnet_depth=2, condition_in_channels=6)
    cn.single_blocks = None  # execute 不使用 single_blocks，释放随机权重
    jt.clean_graph(); jt.gc()
    for k, v in sd.items():
        if not k.startswith(prefix): continue
        nk = k[len(prefix):]
        if nk.startswith('single_blocks'): continue
        parts = nk.split('.')
        # input_hint_block 是 Conv，转 fp32；其余 Linear 权重保持 fp16
        t = v.astype(np.float32) if 'input_hint_block' in nk else v
        setattr(get_mod(cn, parts[:-1]), parts[-1], jt.array(t))
        del t
    jt.clean_graph(); jt.gc()
    return cn
cn1 = load_cn('controlnet1.')
cn2 = load_cn('controlnet2.')
del sd
jt.clean_graph(); jt.gc(); gc.collect()
print(f'CN loaded: {mem()} MiB', flush=True)

# ---- 条件 / 输入 / schedule ----
conds = np.load('/data/PAM/ref/conds.npz')
depth=jt.array(conds['depth']); normal=jt.array(conds['normal'])
hand=jt.array(conds['hand']); seg=jt.array(conds['seg'])
physical=jt.concat([depth,normal],dim=1)
semantic=jt.concat([hand,seg],dim=1)
del conds

raw_inp = {k:v for k,v in np.load('/data/PAM/ref/inp_true.npz').items()}
img_ids=jt.array(raw_inp['img_ids'].astype(np.float32))
txt_ids=jt.array(raw_inp['txt_ids'].astype(np.float32))
txt_raw=jt.array(raw_inp['txt'].astype(np.float32))
y=jt.array(raw_inp['y'].astype(np.float32))
img=jt.array(raw_inp['img'].astype(np.float32))
txt_len = txt_raw.shape[1]

timesteps = np.load('/data/PAM/ref/schedule.npy').astype(np.float64)
ref_steps = np.load('/data/PAM/ref/loop_latents.npz')
g_vec = jt.full((1,),4.0,dtype=jt.float32)
gs=(1.0,1.0)

def main_fwd(img, t_vec, res):
    x=model.img_in(img)
    vec=model.time_in(timestep_embedding(t_vec,256))
    vec=vec+model.guidance_in(timestep_embedding(g_vec,256))
    vec=vec+model.vector_in(y)
    txt=model.txt_in(txt_raw)
    ids=jt.concat((txt_ids,img_ids),dim=1)
    pe=model.pe_embedder(ids)
    cdepth=len(res)
    for ib,blk in enumerate(model.double_blocks):
        x,txt=blk(x,txt,vec,pe)
        x=x+res[ib % cdepth]
        jt.clean_graph(); jt.gc()
    x=jt.concat((txt,x),dim=1); jt.clean_graph(); jt.gc()
    for blk in model.single_blocks:
        x=blk(x,vec,pe); jt.clean_graph(); jt.gc()
    x=x[:,txt_len:,:]
    x=model.final_layer(x,vec)
    return x

print('==== 28-step loop ====', flush=True)
jt_steps=[]
for i in range(28):
    t_curr=float(timesteps[i]); t_prev=float(timesteps[i+1])
    t_vec=jt.full((1,),t_curr,dtype=jt.float32)
    inp=dict(img=img,img_ids=img_ids,txt=txt_raw,txt_ids=txt_ids,y=y)
    r1=cn1(**inp,controlnet_cond=physical,timesteps=t_vec,guidance=g_vec)
    jt.clean_graph(); jt.gc()
    r2=cn2(**inp,controlnet_cond=semantic,timesteps=t_vec,guidance=g_vec)
    jt.clean_graph(); jt.gc()
    res=[a*gs[0]+b*gs[1] for a,b in zip(r1,r2)]
    del r1,r2
    jt.clean_graph(); jt.gc()
    pred=main_fwd(img,t_vec,res)
    img=img+(t_prev-t_curr)*pred
    jt.clean_graph(); jt.gc()
    b=img.numpy().astype(np.float64)
    a=ref_steps[f'step_{i}'].astype(np.float64)
    mse=np.mean((a-b)**2); mx=np.max(np.abs(a-b))
    jt_steps.append(b.astype(np.float32))
    if i%4==0 or i==27:
        print(f'step {i:2d} t={t_curr:.4f} MSE {mse:.4e} MAX {mx:.4e} mem {mem()}',flush=True)

np.save('/data/PAM/ref/jt_final_packed.npy', jt_steps[-1])
# unpack
h,w,c,ph,pw=30,40,16,2,2
xp=jt_steps[-1].reshape(1,h,w,c,ph,pw).transpose(0,3,1,4,2,5).reshape(1,c,h*ph,w*pw)
np.save('/data/PAM/ref/jt_final_unpacked.npy', xp)
ref_u=np.load('/data/PAM/ref/final_unpacked.npy').astype(np.float64)
mse_u=np.mean((ref_u-xp.astype(np.float64))**2)
mx_u=np.max(np.abs(ref_u-xp.astype(np.float64)))
print(f'FINAL unpacked MSE {mse_u:.6e} MAX {mx_u:.6e}', flush=True)
print('DONE', flush=True)
