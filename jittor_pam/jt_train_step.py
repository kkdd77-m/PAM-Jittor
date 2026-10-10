import sys, numpy as np
sys.path.insert(0,'/data/PAM')
import jittor as jt
jt.flags.use_cuda=1; jt.flags.lazy_execution=0
from jittor_pam.controlnet import ControlNetFlux, FluxParams

params = FluxParams(in_channels=64, vec_in_dim=768, context_in_dim=4096,
    hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38,
    axes_dim=[16,56,56], theta=10000, qkv_bias=True, guidance_embed=True)

def get_mod(m, parts):
    for p in parts:
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m

sd = np.load('/data/PAM/ref/controlnet_bf16u16.npz')
def bf16(v):
    return (v.astype(np.uint32)<<16).view(np.float32).reshape(v.shape)
def load_cn(prefix):
    cn = ControlNetFlux(params, controlnet_depth=2, condition_in_channels=6)
    for k,v in sd.items():
        if not k.startswith(prefix): continue
        nk=k[len(prefix):]
        if nk.startswith('single_blocks'): continue
        parts=nk.split('.')
        setattr(get_mod(cn,parts[:-1]), parts[-1], jt.array(bf16(v)))
    jt.clean_graph(); jt.gc()
    return cn
print('loading cn fp32 (bf16-exact)...',flush=True)
cn1=load_cn('controlnet1.'); cn2=load_cn('controlnet2.'); del sd; jt.gc()

B=np.load('/data/PAM/ref/train_batch.npz'); R=np.load('/data/PAM/ref/train_step_ref.npz')
def A(k): return jt.array(B[k].astype(np.float32))
xt,txt,txt_ids,y,img_ids=A('x_t'),A('txt'),A('txt_ids'),A('y'),A('img_ids')
phys=jt.concat([A('depth'),A('normal')],dim=1); sem=jt.concat([A('hand'),A('seg')],dim=1)
t_vec=jt.array([float(B['t'])]); g_vec=jt.array([4.0])
dres=[jt.array(R['dres_%d'%k].astype(np.float32)) for k in range(2)]

r1=cn1(img=xt,img_ids=img_ids,controlnet_cond=phys,txt=txt,txt_ids=txt_ids,timesteps=t_vec,y=y,guidance=g_vec)
r2=cn2(img=xt,img_ids=img_ids,controlnet_cond=sem,txt=txt,txt_ids=txt_ids,timesteps=t_vec,y=y,guidance=g_vec)
proxy=sum((r1[k]*dres[k]).sum()+(r2[k]*dres[k]).sum() for k in range(2))

picks=['controlnet_blocks.0.weight','controlnet_blocks.1.weight','input_hint_block.0.weight',
       'pos_embed_input.weight','img_in.weight','double_blocks.0.img_attn.qkv.weight']
def wvar(cn,name):
    p=name.split('.'); return getattr(get_mod(cn,p[:-1]),p[-1])
vars_=[]; tags=[]
for nm in picks:
    vars_.append(wvar(cn1,nm)); tags.append('cn1_'+nm.replace('.','_'))
    vars_.append(wvar(cn2,nm)); tags.append('cn2_'+nm.replace('.','_'))
grads=jt.grad(proxy, vars_)
print('%-34s %12s %12s %10s'%('param','relL2','cosine','maxAbsDiff'),flush=True)
for tag,g in zip(tags,grads):
    ref=R[tag].astype(np.float64); got=g.numpy().astype(np.float64)
    rel=np.linalg.norm(got-ref)/(np.linalg.norm(ref)+1e-30)
    cos=float((got.ravel()@ref.ravel())/(np.linalg.norm(got)*np.linalg.norm(ref)+1e-30))
    print('%-34s %.4e   %.7f  %.3e'%(tag,rel,cos,np.abs(got-ref).max()),flush=True)
print('DONE')
