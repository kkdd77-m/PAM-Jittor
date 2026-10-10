import sys, torch, numpy as np, json, cv2
torch.backends.cudnn.enabled = False
sys.path.insert(0, '/data/PAM/PAM/appearance_gen')
sys.path.insert(0, '/data/PAM/PAM/appearance_gen/src')
from src.flux.util import load_clip, load_t5, load_ae
from src.flux.sampling import prepare
from einops import rearrange
device='cuda'
prompt = json.loads(open('/data/PAM/PAM/dexycb_prompt.jsonl').readline())['response']

cap = cv2.VideoCapture('/data/PAM/PAM/val_resources/comb_val/val_video.mp4')
ok, frame = cap.read(); cap.release()
frame = cv2.resize(frame, (640,480)); frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = torch.from_numpy(frame.astype(np.float32)/127.5-1.0).permute(2,0,1).unsqueeze(0).to(device)

ae = load_ae('flux-dev', device=device); ae.eval(); ae.reg.sample = False
t5 = load_t5(device); clip = load_clip(device)

with torch.autocast(device_type='cuda', dtype=torch.float32), torch.no_grad():
    x1_4d = ae.encode(img.float())
    inp = prepare(t5=t5, clip=clip, img=x1_4d, prompt=prompt)
    x1 = rearrange(x1_4d, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    g = torch.Generator(device=device).manual_seed(123)
    x0 = torch.randn(x1.shape, generator=g, device=device, dtype=torch.float32)
    t = 0.5
    xt = (1-t)*x1 + t*x0
    target = x0 - x1

conds = np.load('/data/PAM/ref/conds.npz')
np.savez('/data/PAM/ref/train_batch.npz',
    img=img.cpu().numpy(), x1_4d=x1_4d.float().cpu().numpy(),
    x_1=x1.float().cpu().numpy(), x_0=x0.float().cpu().numpy(),
    x_t=xt.float().cpu().numpy(), target=target.float().cpu().numpy(),
    t=np.array(t,np.float32),
    txt=inp['txt'].float().cpu().numpy(), txt_ids=inp['txt_ids'].float().cpu().numpy(),
    y=inp['y'].float().cpu().numpy(), img_ids=inp['img_ids'].float().cpu().numpy(),
    depth=conds['depth'], normal=conds['normal'], hand=conds['hand'], seg=conds['seg'])
print('saved train_batch.npz', tuple(x1.shape), tuple(xt.shape), tuple(target.shape), flush=True)
print('DONE')
