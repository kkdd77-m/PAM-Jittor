import sys, numpy as np, subprocess
sys.path.insert(0,'/data/PAM')
import jittor as jt
jt.flags.use_cuda=1; jt.flags.lazy_execution=0
jt.flags.device_mem_limit=33285996544
def mem():
    return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()
N=int(sys.argv[1]) if len(sys.argv)>1 else 2
LR=float(sys.argv[2]) if len(sys.argv)>2 else 1e-3

_oi=jt.nn.Linear.__init__
def _ti(self,i,o,bias=True):
    self.in_features=i; self.out_features=o
    self.weight=jt.zeros((1,1)); self.bias=jt.zeros((1,)) if bias else None
jt.nn.Linear.__init__=_ti
def _li(self,x):
    w=self.weight.float32()
    if x.dtype!=jt.float32: x=x.float32()
    y=jt.matmul(x,w.transpose())
    if self.bias is not None: y=y+self.bias.float32()
    return y
jt.nn.Linear.execute=_li
from jittor_pam.model import Flux, FluxParams
from jittor_pam.controlnet import ControlNetFlux
from jittor_pam.layers import timestep_embedding
def get_mod(m,p):
    for q in p: m=m[int(q)] if q.isdigit() else getattr(m,q)
    return m
params=FluxParams(in_channels=64,vec_in_dim=768,context_in_dim=4096,hidden_size=3072,mlp_ratio=4.0,
    num_heads=24,depth=19,depth_single_blocks=38,axes_dim=[16,56,56],theta=10000,qkv_bias=True,guidance_embed=True)
print('loading dit fp16...',flush=True)
model=Flux(params)
import json
f=open('/data/PAM/weights/flux_dev/flux1-dev.safetensors','rb')
n=np.frombuffer(f.read(8),np.uint64)[0]; hdr=json.loads(f.read(n)); ds=8+n
for k,v in hdr.items():
    if k=='__metadata__': continue
    p=k.split('.'); mod=get_mod(model,p[:-1]); s0=int(ds+v['data_offsets'][0]); s1=int(ds+v['data_offsets'][1]); shape=v['shape']; d=v['dtype']
    f.seek(s0); raw=f.read(s1-s0)
    if d=='BF16':
        u=np.frombuffer(raw,'<u2').reshape(shape); w=(u.astype(np.uint32)<<16).view(np.float32).astype(np.float16)
    elif d=='F16': w=np.frombuffer(raw,'<f2').reshape(shape).copy()
    else: w=np.frombuffer(raw,'<f4').reshape(shape).astype(np.float16)
    setattr(mod,p[-1],jt.array(w))
f.close(); jt.clean_graph(); jt.gc()
# 保持 tiny-init 构造 CN，避免 fp32 随机权重峰值；setattr 会灌入真实 fp16 权重
print('dit loaded %s MiB'%mem(),flush=True)

csd=np.load('/data/PAM/ref/controlnet_fp16.npz')
def load_cn(prefix):
    cn=ControlNetFlux(params,controlnet_depth=2,condition_in_channels=6)
    for k,v in csd.items():
        if not k.startswith(prefix): continue
        nk=k[len(prefix):]
        if nk.startswith('single_blocks'): continue
        p=nk.split('.')
        t=v.astype(np.float32) if 'input_hint_block' in nk else v
        setattr(get_mod(cn,p[:-1]),p[-1],jt.array(t))
    jt.clean_graph(); jt.gc(); return cn
cn1=load_cn('controlnet1.'); cn2=load_cn('controlnet2.'); del csd; jt.gc()
print('cn loaded %s MiB'%mem(),flush=True)

B=np.load('/data/PAM/ref/train_batch.npz')
def A(k): return jt.array(B[k].astype(np.float32))
xt,txt,txt_ids,y,img_ids,target=A('x_t'),A('txt'),A('txt_ids'),A('y'),A('img_ids'),A('target')
phys=jt.concat([A('depth'),A('normal')],1); sem=jt.concat([A('hand'),A('seg')],1)
t_vec=jt.full((1,),float(B['t'])); g_vec=jt.full((1,),4.0)
txt_len=txt.shape[1]

class CkptDouble(jt.Function):
    def execute(self,blk,im,tx,vec,pe):
        self.blk=blk; self.im=im; self.tx=tx; self.vec=vec; self.pe=pe
        with jt.no_grad():
            o_im,o_tx=blk(im,tx,vec,pe)
        return o_im,o_tx
    def grad(self,g_im,g_tx):
        im=self.im.clone(); tx=self.tx.clone()
        o_im,o_tx=self.blk(im,self.tx,self.vec,self.pe)
        sc=(o_im*g_im).sum()+(o_tx*g_tx).sum()
        gi,gt=jt.grad(sc,[im,tx])
        return None,gi,gt,None,None
class CkptSingle(jt.Function):
    def execute(self,blk,im,vec,pe):
        self.blk=blk; self.im=im; self.vec=vec; self.pe=pe
        with jt.no_grad(): o=blk(im,vec,pe)
        return o
    def grad(self,g):
        im=self.im.clone(); o=self.blk(im,self.vec,self.pe)
        gi=jt.grad((o*g).sum(),im)
        return None,gi,None,None
def dit_fwd(res):
    im=model.img_in(xt)
    vec=model.time_in(timestep_embedding(t_vec,256))+model.guidance_in(timestep_embedding(g_vec,256))+model.vector_in(y)
    tx=model.txt_in(txt); ids=jt.concat((txt_ids,img_ids),1); pe=model.pe_embedder(ids)
    cd=len(res)
    for i,blk in enumerate(model.double_blocks):
        im,tx=CkptDouble.apply(blk,im,tx,vec,pe); im=im+res[i%cd]
    im=jt.concat((tx,im),1)
    for blk in model.single_blocks: im=CkptSingle.apply(blk,im,vec,pe)
    im=im[:,txt_len:,:]; return model.final_layer(im,vec)

trainable=list(cn1.parameters())+list(cn2.parameters())
print('trainable params %.1fM'%(sum(t.numel() for t in trainable)/1e6),flush=True)
opt=jt.nn.Adam(trainable,lr=LR,eps=1e-8)
inp=dict(img=xt,img_ids=img_ids,txt=txt,txt_ids=txt_ids,y=y)
for it in range(N):
    r1=cn1(**inp,controlnet_cond=phys,timesteps=t_vec,guidance=g_vec)
    r2=cn2(**inp,controlnet_cond=sem,timesteps=t_vec,guidance=g_vec)
    res=[a+b for a,b in zip(r1,r2)]
    pred=dit_fwd(res)
    loss=((pred-target)**2).mean()
    lv=float(loss.numpy())
    opt.zero_grad(); opt.step(loss)
    print('iter %3d  loss %.6e  mem %s'%(it,lv,mem()),flush=True)
    jt.clean_graph(); jt.gc()
print('DONE')
