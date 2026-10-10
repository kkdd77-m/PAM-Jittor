import sys, torch, numpy as np
torch.backends.cudnn.enabled = False
sys.path.insert(0, '/data/PAM/PAM/appearance_gen')
sys.path.insert(0, '/data/PAM/PAM/appearance_gen/src')
from src.flux.util import load_flow_model
from src.flux.sampling import get_schedule
device = 'cuda'
model = load_flow_model('flux-dev', device=device)
model.eval()
inp = {k: torch.from_numpy(v).to(device) for k, v in np.load('/data/PAM/ref/inp_true.npz').items()}
timesteps = get_schedule(28, (640//8)*(480//8)//(16*16), shift=True)
t_curr = timesteps[0]
t_vec = torch.full((1,), t_curr, dtype=torch.float32, device=device)
guidance_vec = torch.full((1,), 4.0, dtype=torch.float32, device=device)
with torch.autocast(device_type='cuda', dtype=torch.float32), torch.no_grad():
    pred = model(**inp, timesteps=t_vec, guidance=guidance_vec)
np.save('/data/PAM/ref/main_pred.npy', pred.float().cpu().numpy())
print('saved main_pred.npy', pred.shape, flush=True)
print('DONE')
