import numpy as np, jittor as jt
jt.flags.use_cuda=1; jt.flags.lazy_execution=0
import sys; sys.path.insert(0,'/data/PAM/jittor_pam')
from autoencoder import AutoEncoder, AutoEncoderParams
from safetensors import safe_open

params = AutoEncoderParams(resolution=1024, in_channels=3, ch=128, out_ch=3,
    ch_mult=[1,2,4,4], num_res_blocks=2, z_channels=16,
    scale_factor=0.3611, shift_factor=0.1159)
vae = AutoEncoder(params); vae.reg.sample=False

sd={}
with safe_open('/data/PAM/weights/flux_dev/ae.safetensors', framework='np', device='cpu') as f:
    for k in f.keys(): sd[k]=f.get_tensor(k)
def load(sub):
    d={k.replace(sub+'.',''): jt.array(v.astype(np.float32)) for k,v in sd.items() if k.startswith(sub+'.')}
    getattr(vae,sub).load_state_dict(d); print(sub,'keys',len(d),flush=True)
load('encoder'); load('decoder')

B=np.load('/data/PAM/ref/train_batch.npz')
def stat(name,a,b):
    a=a.astype(np.float64); b=b.astype(np.float64)
    print('%-10s MSE %.4e  MAX %.4e'% (name,((a-b)**2).mean(),np.abs(a-b).max()),flush=True)

img=jt.array(B['img'].astype(np.float32))
with jt.no_grad():
    z=vae.encode(img)                      # [1,16,60,80]
z_np=z.numpy()
stat('x1_4d', B['x1_4d'], z_np)

# pack: b c (h ph) (w pw) -> b (h w) (c ph pw), ph=pw=2
b,c,h,w=z.shape
x1=z.reshape(b,c,h//2,2,w//2,2).transpose(0,2,4,1,3,5).reshape(b,(h//2)*(w//2),c*4)
stat('x_1', B['x_1'], x1.numpy())

x0=jt.array(B['x_0'].astype(np.float32)); t=float(B['t'])
xt=(1-t)*x1+t*x0
target=x0-x1
stat('x_t', B['x_t'], xt.numpy())
stat('target', B['target'], target.numpy())
print('DONE')
