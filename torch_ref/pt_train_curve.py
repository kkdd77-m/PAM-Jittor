import sys, numpy as np, torch
torch.backends.cudnn.enabled = False
sys.path.insert(0,'/data/PAM/PAM/appearance_gen'); sys.path.insert(0,'/data/PAM/PAM/appearance_gen/src')
from src.flux.model import Flux
from src.flux.controlnet import FluxParams, ControlNetFlux, zero_module
from src.flux.modules.layers import timestep_embedding
from safetensors.torch import load_file

dev="cuda"; K=200; N=20; LR=3e-4
params=FluxParams(in_channels=64,vec_in_dim=768,context_in_dim=4096,hidden_size=3072,mlp_ratio=4.0,
    num_heads=24,depth=2,depth_single_blocks=2,axes_dim=[16,56,56],theta=10_000,qkv_bias=True,guidance_embed=True)

print('loading dit (depth=2)...',flush=True)
dit=Flux(params)
sd=load_file('/data/PAM/weights/flux_dev/flux1-dev.safetensors')
m,u=dit.load_state_dict(sd,strict=False,assign=True); del sd
dit=dit.float().to(dev).eval(); dit.requires_grad_(False)

torch.manual_seed(0)
cn=ControlNetFlux(params,controlnet_depth=2,condition_in_channels=6)
cn.load_state_dict(dit.state_dict(),strict=False)
cn.double_blocks=dit.double_blocks              # 与 Jittor 一致：共享
zero_module(cn.pos_embed_input)                 # Jittor 中为零
cn=cn.float().to(dev); cn.train()
# 冻结非训练部分
for nm,p in cn.named_parameters():
    p.requires_grad = ('input_hint_block' in nm or 'pos_embed_input' in nm or 'controlnet_blocks' in nm)

# 导出确定性 input_hint 初始化（供 Jittor 加载，保证两边起点一致）
init={k:v.detach().float().cpu().numpy() for k,v in cn.input_hint_block.state_dict().items()}
np.savez('/data/PAM/ref/cn_init.npz',**init)
print('saved cn_init.npz',list(init.keys()),flush=True)

B=np.load('/data/PAM/ref/train_batch.npz')
def T(k): return torch.from_numpy(B[k]).float().to(dev)
xt,txt,txt_ids,y,img_ids,target=T('x_t'),T('txt'),T('txt_ids'),T('y'),T('img_ids'),T('target')
phys=torch.cat([T('depth'),T('normal')],1)
xt=xt[:,:K,:]; target=target[:,:K,:]; img_ids=img_ids[:,:K,:]; phys=phys[:,:,:80,:]
t_vec=torch.full((1,),float(B['t']),device=dev); g_vec=torch.full((1,),4.0,device=dev)
txt_len=txt.shape[1]
inp=dict(img=xt,img_ids=img_ids,txt=txt,txt_ids=txt_ids,y=y)

trainable=[p for n,p in cn.named_parameters() if p.requires_grad]
print('trainable %.1fM'%(sum(p.numel() for p in trainable)/1e6),flush=True)
opt=torch.optim.Adam(trainable,lr=LR,eps=1e-8)

def dit_fwd(res):
    with torch.no_grad():
        im=dit.img_in(xt)
        vec=dit.time_in(timestep_embedding(t_vec,256))+dit.guidance_in(timestep_embedding(g_vec,256))+dit.vector_in(y)
        tx=dit.txt_in(txt); ids=torch.cat((txt_ids,img_ids),1); pe=dit.pe_embedder(ids)
    for i,blk in enumerate(dit.double_blocks):
        im,tx=blk(img=im,txt=tx,vec=vec,pe=pe,image_proj=None,ip_scale=1.0)
        im=im+res[i%len(res)]
    im=torch.cat((tx,im),1)
    for blk in dit.single_blocks: im=blk(im,vec=vec,pe=pe)
    im=im[:,txt_len:,:]
    return dit.final_layer(im,vec)

losses=[]
for it in range(N):
    res=cn(**inp,controlnet_cond=phys,timesteps=t_vec,guidance=g_vec)
    pred=dit_fwd(res)
    loss=((pred.float()-target)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    lv=float(loss.item()); losses.append(lv)
    print('iter %3d  loss %.6e'%(it,lv),flush=True)
np.save('/data/PAM/ref/pt_loss_curve.npy',np.array(losses))
print('DONE first %.4f last %.4f'%(losses[0],losses[-1]),flush=True)
