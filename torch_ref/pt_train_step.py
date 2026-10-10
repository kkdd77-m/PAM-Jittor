import sys, torch, numpy as np
torch.backends.cudnn.enabled = False
sys.path.insert(0,'/data/PAM/PAM/appearance_gen'); sys.path.insert(0,'/data/PAM/PAM/appearance_gen/src')
from src.flux.util import load_flow_model, load_controlnet_trained
from src.flux.modules.layers import timestep_embedding
from torch.utils.checkpoint import checkpoint
dev='cuda'
B=np.load('/data/PAM/ref/train_batch.npz')
def T(k): return torch.from_numpy(B[k]).to(dev)
xt,txt,txt_ids,y,img_ids,target=T('x_t'),T('txt'),T('txt_ids'),T('y'),T('img_ids'),T('target')
physical=torch.cat([T('depth'),T('normal')],1); semantic=torch.cat([T('hand'),T('seg')],1)
t_vec=torch.full((1,),float(B['t']),device=dev); g_vec=torch.full((1,),4.0,device=dev)
inp=dict(img=xt,txt=txt,txt_ids=txt_ids,y=y,img_ids=img_ids)

def load_cn():
    sd=torch.load('/data/PAM/weights/pam_ckpt/appearance_gen/controlnet.bin',map_location='cpu',weights_only=True)
    c1={k.replace('controlnet1.',''):v for k,v in sd.items() if k.startswith('controlnet1.')}
    c2={k.replace('controlnet2.',''):v for k,v in sd.items() if k.startswith('controlnet2.')}
    del sd
    a=load_controlnet_trained('flux-dev',dev,c1,6,2)
    b=load_controlnet_trained('flux-dev',dev,c2,6,2)
    return a,b

# ---- Stage A: residuals (no_grad, bf16 cn) then dit checkpoint backward ----
print('loading dit+cn...',flush=True)
dit=load_flow_model('flux-dev',device=dev); dit.eval(); dit.requires_grad_(False)
cn1,cn2=load_cn(); cn1.eval(); cn2.eval()
with torch.autocast('cuda',dtype=torch.float32), torch.no_grad():
    r1=cn1(**inp,controlnet_cond=physical,timesteps=t_vec,guidance=g_vec)
    r2=cn2(**inp,controlnet_cond=semantic,timesteps=t_vec,guidance=g_vec)
res=[(a.detach().float()+b.detach().float()).requires_grad_(True) for a,b in zip(r1,r2)]
del cn1,cn2,r1,r2; torch.cuda.empty_cache()
print('stageA dit fwd (checkpoint)... peak %.1f'%(torch.cuda.memory_allocated()/1e9),flush=True)
with torch.autocast('cuda',dtype=torch.float32):
    with torch.no_grad():
        im=dit.img_in(xt)
        vec=dit.time_in(timestep_embedding(t_vec,256))+dit.guidance_in(timestep_embedding(g_vec,256))+dit.vector_in(y)
        tx=dit.txt_in(txt); ids=torch.cat((txt_ids,img_ids),1); pe=dit.pe_embedder(ids)
    def dblk(b):
        def f(im_,tx_): return b(img=im_,txt=tx_,vec=vec,pe=pe,image_proj=None,ip_scale=1.0)
        return f
    def sblk(b):
        def f(im_): return b(im_,vec=vec,pe=pe)
        return f
    cd=len(res)
    for i,blk in enumerate(dit.double_blocks):
        im,tx=checkpoint(dblk(blk),im,tx,use_reentrant=False)
        im=im+res[i%cd]
    im=torch.cat((tx,im),1)
    for blk in dit.single_blocks:
        im=checkpoint(sblk(blk),im,use_reentrant=False)
    im=im[:,tx.shape[1]:,...]
    pred=dit.final_layer(im,vec)
    loss=torch.nn.functional.mse_loss(pred.float(),target.float())
loss.backward()
dres=[q.grad.detach().float().cpu() for q in res]
print('loss %.8e  peak %.1fGB'%(loss.item(),torch.cuda.max_memory_allocated()/1e9),flush=True)
del dit,pred,im; torch.cuda.empty_cache()

# ---- Stage B: fp32 cn backward with upstream dres ----
print('stageB cn fp32 backward...',flush=True)
cn1,cn2=load_cn(); cn1.float().train(); cn2.float().train()
with torch.autocast('cuda',dtype=torch.float32):
    r1=cn1(**inp,controlnet_cond=physical,timesteps=t_vec,guidance=g_vec)
    r2=cn2(**inp,controlnet_cond=semantic,timesteps=t_vec,guidance=g_vec)
    proxy=sum((r1[k]*dres[k].to(dev)).sum()+(r2[k]*dres[k].to(dev)).sum() for k in range(2))
proxy.backward()
print('stageB peak %.1fGB'%(torch.cuda.max_memory_allocated()/1e9),flush=True)
pick=['controlnet_blocks.0.weight','controlnet_blocks.1.weight','input_hint_block.0.weight',
      'pos_embed_input.weight','img_in.weight','double_blocks.0.img_attn.qkv.weight']
out={'loss':np.array(loss.item())}
for k in range(2): out['dres_%d'%k]=dres[k].numpy()
for nm in pick:
    tag=nm.replace('.','_')
    out['cn1_'+tag]=dict(cn1.named_parameters())[nm].grad.float().cpu().numpy()
    out['cn2_'+tag]=dict(cn2.named_parameters())[nm].grad.float().cpu().numpy()
np.savez('/data/PAM/ref/train_step_ref.npz',**out)
print('saved train_step_ref.npz DONE',flush=True)
