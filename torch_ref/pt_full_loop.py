import sys, torch, numpy as np
torch.backends.cudnn.enabled = False
sys.path.insert(0, '/data/PAM/PAM/appearance_gen')
sys.path.insert(0, '/data/PAM/PAM/appearance_gen/src')
from src.flux.util import load_flow_model, load_controlnet_trained
from src.flux.sampling import get_schedule, unpack
device = 'cuda'

print('loading main model...', flush=True)
model = load_flow_model('flux-dev', device=device); model.eval()

print('loading controlnets...', flush=True)
sd = torch.load('/data/PAM/weights/pam_ckpt/appearance_gen/controlnet.bin',
                map_location='cpu', weights_only=True)
c1 = {k.replace('controlnet1.',''):v for k,v in sd.items() if k.startswith('controlnet1.')}
c2 = {k.replace('controlnet2.',''):v for k,v in sd.items() if k.startswith('controlnet2.')}
del sd
cn1 = load_controlnet_trained('flux-dev', device, c1, 6, 2); cn1.eval()
cn2 = load_controlnet_trained('flux-dev', device, c2, 6, 2); cn2.eval()

conds = np.load('/data/PAM/ref/conds.npz')
depth  = torch.from_numpy(conds['depth']).to(device)
normal = torch.from_numpy(conds['normal']).to(device)
hand   = torch.from_numpy(conds['hand']).to(device)
seg    = torch.from_numpy(conds['seg']).to(device)
physical = torch.cat([depth, normal], dim=1)
semantic = torch.cat([hand, seg], dim=1)

inp = {k: torch.from_numpy(v).to(device) for k,v in np.load('/data/PAM/ref/inp_true.npz').items()}
H, W = 480, 640
timesteps = get_schedule(28, (W//8)*(H//8)//(16*16), shift=True)
np.save('/data/PAM/ref/schedule.npy', np.array(timesteps, dtype=np.float64))
guidance = 4.0
gs = (1.0, 1.0, 0.0, 0.0)

img = inp['img']
guidance_vec = torch.full((1,), guidance, device=device, dtype=img.dtype)
steps = []
with torch.autocast(device_type='cuda', dtype=torch.float32), torch.no_grad():
    for i,(t_curr,t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((1,), t_curr, dtype=img.dtype, device=device)
        inp['img'] = img
        r1 = cn1(**inp, controlnet_cond=physical, timesteps=t_vec, guidance=guidance_vec)
        r2 = cn2(**inp, controlnet_cond=semantic, timesteps=t_vec, guidance=guidance_vec)
        res = [a*gs[0] + b*gs[1] for a,b in zip(r1,r2)]
        pred = model(**inp, timesteps=t_vec, guidance=guidance_vec,
                     block_controlnet_hidden_states=res)
        img = img + (t_prev - t_curr) * pred
        steps.append(img.float().cpu().numpy())
        if i % 4 == 0 or i == 27:
            print(f'step {i:2d} t={t_curr:.4f} peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB', flush=True)

np.savez('/data/PAM/ref/loop_latents.npz',
         **{f'step_{i}': s for i,s in enumerate(steps)})
np.save('/data/PAM/ref/final_packed.npy', steps[-1])
x0 = unpack(torch.from_numpy(steps[-1]).to(device).float(), H, W)
np.save('/data/PAM/ref/final_unpacked.npy', x0.float().cpu().numpy())
print('saved final', steps[-1].shape, x0.shape, flush=True)
print('DONE', flush=True)
