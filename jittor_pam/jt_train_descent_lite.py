import sys, numpy as np, subprocess
sys.path.insert(0,'/data/PAM')
import jittor as jt
jt.flags.use_cuda=1; jt.flags.lazy_execution=0
jt.flags.device_mem_limit=33285996544
def mem():
    return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()
N=int(sys.argv[1]) if len(sys.argv)>1 else 24
LR=float(sys.argv[2]) if len(sys.argv)>2 else 5e-3
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
import json
def get_mod(m,p):
    for q in p: m=m[int(q)] if q.isdigit() else getattr(m,q)
    return m
class Id(jt.Module):
    def execute(self,x): return x
params=FluxParams(in_channels=64,vec_in_dim=768,context_in_dim=4096,hidden_size=3072,mlp_ratio=4.0,
    num_heads=24,depth=19,depth_single_blocks=38,axes_dim=[16,56,56],theta=10000,qkv_bias=True,guidance_embed=True)
model=Flux(params)
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
csd=np.load('/data/PAM/ref/controlnet_fp16.npz')
def load_cn(prefix):
    cn=ControlNetFlux(params,controlnet_depth=2,condition_in_channels=6)
    for k,v in csd.items():
        if not k.startswith(prefix): continue
        nk=k[len(prefix):]
        if nk.startswith('single_blocks'): continue
        pp=nk.split('.')
        t=v.astype(np.float32) if 'input_hint_block' in nk else v
        setattr(get_mod(cn,pp[:-1]),pp[-1],jt.array(t))
    jt.clean_graph(); jt.gc(); return cn
cn1=load_cn('controlnet1.'); cn2=load_cn('controlnet2.'); del csd; jt.gc()
print('weights loaded %s MiB'%mem(),flush=True)
B=np.load('/data/PAM/ref/train_batch.npz')
def A(k): return jt.array(B[k].astype(np.float32))
xt,txt,txt_ids,y,img_ids,target=A('x_t'),A('txt'),A('txt_ids'),A('y'),A('img_ids'),A('target')
phys=jt.concat([A('depth'),A('normal')],1); sem=jt.concat([A('hand'),A('seg')],1)
t_vec=jt.full((1,),float(B['t'])); g_vec=jt.full((1,),4.0); txt_len=txt.shape[1]
# 预计算全分辨率条件 CNN 特征（no_grad），随后替换为 Identity，训练时不保留全分辨率激活
with jt.no_grad():
    feat1=cn1.input_hint_block(phys); feat2=cn2.input_hint_block(sem)
cn1.input_hint_block=Id(); cn2.input_hint_block=Id()
jt.clean_graph(); jt.gc()
class CkptDouble(jt.Function):
    def execute(self,blk,im,tx,vec,pe):
        self.blk=blk; self.im=im; self.tx=tx; self.vec=vec; self.pe=pe
        with jt.no_grad(): a,b=blk(im,tx,vec,pe)
        return a,b
    def grad(self,ga,gb):
        im=self.im.clone(); tx=self.tx.clone(); a,b=self.blk(im,self.tx,self.vec,self.pe)
        gi,gt=jt.grad((a*ga).sum()+(b*gb).sum(),[im,tx]); return None,gi,gt,None,None
class CkptSingle(jt.Function):
    def execute(self,blk,im,vec,pe):
        self.blk=blk; self.im=im; self.vec=vec; self.pe=pe
        with jt.no_grad(): return blk(im,vec,pe)
    def grad(self,g):
        im=self.im.clone(); o=self.blk(im,self.vec,self.pe)
        return None,jt.grad((o*g).sum(),im),None,None
def dit_fwd(res):
    im=model.img_in(xt)
    vec=model.time_in(timestep_embedding(t_vec,256))+model.guidance_in(timestep_embedding(g_vec,256))+model.vector_in(y)
    tx=model.txt_in(txt); pe=model.pe_embedder(jt.concat((txt_ids,img_ids),1)); cd=len(res)
    for i,blk in enumerate(model.double_blocks):
        im,tx=CkptDouble.apply(blk,im,tx,vec,pe); im=im+res[i%cd]
    im=jt.concat((tx,im),1)
    for blk in model.single_blocks: im=CkptSingle.apply(blk,im,vec,pe)
    return model.final_layer(im[:,txt_len:,:],vec)
def adapters(cn):
    v=[]
    for cb in cn.controlnet_blocks: v+= [cb.weight,cb.bias]
    v+= [cn.pos_embed_input.weight,cn.pos_embed_input.bias]
    return v
tv=adapters(cn1)+adapters(cn2)
print('trainable adapter params %.1fM'%(sum(t.numel() for t in tv)/1e6),flush=True)
opt=jt.nn.SGD(tv,lr=LR)
inp=dict(img=xt,img_ids=img_ids,txt=txt,txt_ids=txt_ids,y=y)
for it in range(N):
    r1=cn1(**inp,controlnet_cond=feat1,timesteps=t_vec,guidance=g_vec)
    r2=cn2(**inp,controlnet_cond=feat2,timesteps=t_vec,guidance=g_vec)
    res=[a+b for a,b in zip(r1,r2)]
    pred=dit_fwd(res); loss=((pred-target)**2).mean(); lv=float(loss.numpy())
    opt.zero_grad(); opt.step(loss)
    print('iter %3d  loss %.6e  mem %s'%(it,lv,mem()),flush=True)
    jt.clean_graph(); jt.gc()
print('DONE')
