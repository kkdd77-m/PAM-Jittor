import sys, numpy as np, subprocess
sys.path.insert(0,'/data/PAM')
import jittor as jt
jt.flags.use_cuda=1; jt.flags.lazy_execution=0
jt.flags.device_mem_limit=74*1024**3
def mem():
    return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip()
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
LR=float(sys.argv[2]) if len(sys.argv)>2 else 1e-3
def _ti(self,i,o,bias=True):
    self.in_features=i; self.out_features=o
    self.weight=jt.zeros((1,1)); self.bias=jt.zeros((1,)) if bias else None
jt.nn.Linear.__init__=_ti
def _li(self,x):
    x=x.float32(); w=self.weight
    if w.dtype!=jt.float32: w=w.float32()
    y=jt.matmul(x,w.transpose(1,0))
    if self.bias is not None:
        b=self.bias
        if b.dtype!=jt.float32: b=b.float32()
        y=y+b
    return y
TRAIN_IDS=set()
jt.nn.Linear.execute=_li
from jittor_pam.model import Flux, FluxParams
from jittor_pam.controlnet import ControlNetFlux
from jittor_pam.layers import timestep_embedding, SafeLinearFn
import json
def get_mod(m,p):
    for q in p: m=m[int(q)] if q.isdigit() else getattr(m,q)
    return m
params=FluxParams(in_channels=64,vec_in_dim=768,context_in_dim=4096,hidden_size=3072,mlp_ratio=4.0,
    num_heads=24,depth=2,depth_single_blocks=2,axes_dim=[16,56,56],theta=1000,qkv_bias=True,guidance_embed=True)
print('loading dit...',flush=True)
model=Flux(params)
f=open('/data/PAM/weights/flux_dev/flux1-dev.safetensors','rb')
n=np.frombuffer(f.read(8),np.uint64)[0]; hdr=json.loads(f.read(n)); ds=8+n
for k,v in hdr.items():
    if k=='__metadata__': continue
    p=k.split('.')
    try:
        mod=get_mod(model,p[:-1])
    except Exception:
        continue
    s0=int(ds+v['data_offsets'][0]); s1=int(ds+v['data_offsets'][1]); shape=v['shape']; d=v['dtype']
    f.seek(s0); raw=f.read(s1-s0)
    if d=='BF16':
        u=np.frombuffer(raw,'<u2').reshape(shape); w=(u.astype(np.uint32)<<16).view(np.float32)
    elif d=='F16': w=np.frombuffer(raw,'<f2').reshape(shape).astype(np.float32)
    else: w=np.frombuffer(raw,'<f4').reshape(shape)
    setattr(mod,p[-1],jt.array(w))
f.close(); jt.clean_graph(); jt.gc()
print('dit loaded %s MiB'%mem(),flush=True)
cn=ControlNetFlux(params,controlnet_depth=2,condition_in_channels=6)
sd=dict(model.named_parameters()); cd=dict(cn.named_parameters())
for k,v in sd.items():
    if k in cd: cd[k].assign(v)
cn.pos_embed_input.weight=jt.zeros((3072,64)); cn.pos_embed_input.bias=jt.zeros((3072,))
for _L in cn.controlnet_blocks:
    _L.weight=jt.zeros((3072,3072)); _L.bias=jt.zeros((3072,))
cn.double_blocks = model.double_blocks
for _m in [cn.img_in,cn.time_in,cn.vector_in,cn.guidance_in,cn.txt_in]:
    for _p in _m.parameters(): _p.stop_grad()
# 关键省显存：所有冻结权重 stop_grad，不分配梯度缓冲（梯度仍正常流过）
for _p in model.parameters(): _p.stop_grad()
import jittor as _jt
for _nm,_p in cn.named_parameters():
    if not ('input_hint_block' in _nm or 'pos_embed_input' in _nm or 'controlnet_blocks' in _nm):
        _p.stop_grad()
print('frozen grads disabled',flush=True)
for _nm,_p in cn.named_parameters():
    if ('input_hint_block' in _nm or 'pos_embed_input' in _nm or 'controlnet_blocks' in _nm):
        TRAIN_IDS.add(id(_p))
print('train ids',len(TRAIN_IDS),flush=True)
print('cn built %s MiB'%mem(),flush=True)
_init=np.load('/data/PAM/ref/cn_init.npz')
_sd=dict(cn.input_hint_block.named_parameters())
for _k in _init.files:
    _sd[_k].assign(jt.array(_init[_k].astype(np.float32)))
print('input_hint init synced from torch',flush=True)
B=np.load('/data/PAM/ref/train_batch.npz')
def A(k): return jt.array(B[k].astype(np.float32))
xt,txt,txt_ids,y,img_ids,target=A('x_t'),A('txt'),A('txt_ids'),A('y'),A('img_ids'),A('target')
phys=jt.concat([A('depth'),A('normal')],1)
# 少量数据演示：取前15行 latent(15x40=600 token)，条件图同步裁，fp32注意力可装入80G
K=200
xt=xt[:,:K,:]; target=target[:,:K,:]; img_ids=img_ids[:,:K,:]
phys=phys[:,:,:80,:]
print('sliced %d img tokens'%K,flush=True)
t_vec=jt.full((1,),float(B['t'])); g_vec=jt.full((1,),4.0)
txt_len=txt.shape[1]
class CkptDouble(jt.Function):
    def execute(self,blk,im,tx,vec,pe):
        self.blk=blk; self.im=im; self.tx=tx; self.vec=vec; self.pe=pe
        with jt.no_grad(): o_im,o_tx=blk(im,tx,vec,pe)
        return o_im,o_tx
    def grad(self,g_im,g_tx):
        g_im=g_im.float16(); g_tx=g_tx.float16()
        im=self.im.clone(); tx=self.tx.clone()
        o_im,o_tx=self.blk(im,tx,self.vec,self.pe)
        sc=(o_im*g_im).sum()+(o_tx*g_tx).sum()
        gi,gt=jt.grad(sc,[im,tx])
        return None,gi,gt,None,None
class CkptSingle(jt.Function):
    def execute(self,blk,im,vec,pe):
        self.blk=blk; self.im=im; self.vec=vec; self.pe=pe
        with jt.no_grad(): o=blk(im,vec,pe)
        return o
    def grad(self,g):
        g=g.float16()
        im=self.im.clone(); o=self.blk(im,self.vec,self.pe)
        print('SG',o.dtype,g.dtype,im.dtype,flush=True)
        gi=jt.grad((o*g).sum(),im)
        return None,gi,None,None
def dit_fwd(res):
    im=model.img_in(xt)
    vec=model.time_in(timestep_embedding(t_vec,256))+model.guidance_in(timestep_embedding(g_vec,256))+model.vector_in(y)
    tx=model.txt_in(txt); ids=jt.concat((txt_ids,img_ids),1); pe=model.pe_embedder(ids)
    for i,blk in enumerate(model.double_blocks):
        im,tx=CkptDouble.apply(blk,im,tx,vec,pe); im=im+res[i%len(res)]
    im=jt.concat((tx,im),1)
    for blk in model.single_blocks: im=CkptSingle.apply(blk,im,vec,pe)
    im=im[:,txt_len:,:]; return model.final_layer(im,vec)
trainable=list(cn.input_hint_block.parameters())+list(cn.pos_embed_input.parameters())
for L in cn.controlnet_blocks: trainable+=list(L.parameters())
print('trainable %.1fM'%(sum(t.numel() for t in trainable)/1e6),flush=True)
opt=jt.nn.Adam(trainable,lr=LR,eps=1e-8)
inp=dict(img=xt,img_ids=img_ids,txt=txt,txt_ids=txt_ids,y=y)
losses=[]
for it in range(N):
    res=cn(**inp,controlnet_cond=phys,timesteps=t_vec,guidance=g_vec)
    pred=dit_fwd(res)
    loss=((pred.float32()-target)**2).mean()
    lv=float(loss.numpy())
    opt.zero_grad(); opt.step(loss)
    losses.append(lv)
    print('iter %3d  loss %.6e  mem %s'%(it,lv,mem()),flush=True)
    jt.clean_graph(); jt.gc()
np.save('/data/PAM/ref/a800_loss_curve.npy', np.array(losses))
print('DONE first %.4f last %.4f'%(losses[0],losses[-1]),flush=True)
