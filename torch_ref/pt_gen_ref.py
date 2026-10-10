import sys, torch, numpy as np, json
torch.backends.cudnn.enabled = False
sys.path.insert(0, '/data/PAM/PAM/appearance_gen')
sys.path.insert(0, '/data/PAM/PAM/appearance_gen/src')
from src.flux.util import load_clip, load_t5
from src.flux.sampling import prepare, get_noise
device = 'cuda'
prompt = json.loads(open('/data/PAM/PAM/dexycb_prompt.jsonl').readline())['response']
print('T5/CLIP encode...', flush=True)
t5 = load_t5(device)
clip = load_clip(device)
x = get_noise(1, 480, 640, device=device, dtype=torch.float32, seed=42)
with torch.no_grad():
    inp_true = prepare(t5=t5, clip=clip, img=x, prompt=prompt)
np.savez('/data/PAM/ref/inp_true.npz', **{k: v.cpu().float().numpy() for k, v in inp_true.items()})
np.save('/data/PAM/ref/t_curr.npy', np.array(1.0, dtype=np.float32))
print('saved inp_true.npz + t_curr.npy', flush=True)
del t5, clip, inp_true
torch.cuda.empty_cache()
print('now main step...', flush=True)
