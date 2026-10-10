import sys, torch, numpy as np, cv2
torch.backends.cudnn.enabled = False
sys.path.insert(0, '/data/PAM/PAM/appearance_gen')
sys.path.insert(0, '/data/PAM/PAM/appearance_gen/src')
from src.flux.util import load_ae
device='cuda'
ae = load_ae('flux-dev', device=device); ae.eval()
z = torch.from_numpy(np.load('/data/PAM/ref/final_unpacked.npy')).to(device).float()
with torch.autocast(device_type='cuda', dtype=torch.float32), torch.no_grad():
    x = ae.decode(z).float().cpu().numpy()
np.save('/data/PAM/ref/pt_decoded.npy', x)
img = (127.5*(np.clip(x[0].transpose(1,2,0),-1,1)+1)).astype(np.uint8)
cv2.imwrite('/data/PAM/ref/pt_final.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print('pt decoded', x.shape, 'min %.3f max %.3f' % (x.min(), x.max()), flush=True)
print('DONE')
