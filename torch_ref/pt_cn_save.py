import sys, torch, cv2, numpy as np
torch.backends.cudnn.enabled = False
sys.path.insert(0, '/data/PAM/PAM/appearance_gen')
sys.path.insert(0, '/data/PAM/PAM/appearance_gen/src')
from src.flux.util import load_controlnet_trained
from src.flux.sampling import get_schedule
device = 'cuda'
sd = torch.load('/data/PAM/weights/pam_ckpt/appearance_gen/controlnet.bin', map_location='cpu', weights_only=True)
np.savez('/data/PAM/ref/controlnet_fp32.npz', **{k: v.float().numpy() for k, v in sd.items()})
print('saved controlnet_fp32.npz')
c1 = {k.replace('controlnet1.', ''): v for k, v in sd.items() if k.startswith('controlnet1.')}
c2 = {k.replace('controlnet2.', ''): v for k, v in sd.items() if k.startswith('controlnet2.')}
cn1 = load_controlnet_trained('flux-dev', device, c1, 6, 2)
cn2 = load_controlnet_trained('flux-dev', device, c2, 6, 2)
cn1.eval(); cn2.eval()
print('CN loaded')
base = '/data/PAM/PAM/val_resources/comb_val'
def load_cond(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame.astype(np.float32)/127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
depth = load_cond(f'{base}/val_depth.mp4')
normal = load_cond(f'{base}/val_normal.mp4')
hand = load_cond(f'{base}/val_hand_keypoints.mp4')
seg = load_cond(f'{base}/val_seg_mask.mp4')
np.savez('/data/PAM/ref/conds.npz', depth=depth.cpu().numpy(), normal=normal.cpu().numpy(),
         hand=hand.cpu().numpy(), seg=seg.cpu().numpy())
print('saved conds.npz')
physical = torch.cat([depth, normal], dim=1)
semantic = torch.cat([hand, seg], dim=1)
inp = {k: torch.from_numpy(v).to(device) for k, v in np.load('/data/PAM/ref/inp_true.npz').items()}
timesteps = get_schedule(28, (640//8)*(480//8)//(16*16), shift=True)
t_curr = timesteps[0]
np.save('/data/PAM/ref/t_curr.npy', np.array(t_curr, dtype=np.float32))
t_vec = torch.full((1,), t_curr, dtype=torch.float32, device=device)
guidance_vec = torch.full((1,), 4.0, dtype=torch.float32, device=device)
print('t_curr =', t_curr, 'input ready', flush=True)
with torch.autocast(device_type='cuda', dtype=torch.float32), torch.no_grad():
    r1 = cn1(**inp, controlnet_cond=physical, timesteps=t_vec, guidance=guidance_vec)
    r2 = cn2(**inp, controlnet_cond=semantic, timesteps=t_vec, guidance=guidance_vec)
np.savez('/data/PAM/ref/cn_out.npz',
         cn1_0=r1[0].float().cpu().numpy(), cn1_1=r1[1].float().cpu().numpy(),
         cn2_0=r2[0].float().cpu().numpy(), cn2_1=r2[1].float().cpu().numpy())
print('saved cn_out.npz', r1[0].shape, r2[0].shape, flush=True)
print('DONE')
