import sys, numpy as np, cv2, jittor as jt
sys.path.insert(0, '/data/PAM')
from jittor_pam.autoencoder import AutoEncoder, AutoEncoderParams
from safetensors import safe_open
params = AutoEncoderParams(resolution=1024, in_channels=3, ch=128, out_ch=3,
    ch_mult=[1,2,4,4], num_res_blocks=2, z_channels=16,
    scale_factor=0.3611, shift_factor=0.1159)
vae = AutoEncoder(params)
sd = {}
with safe_open('/data/PAM/weights/flux_dev/ae.safetensors', framework='np', device='cpu') as f:
    for k in f.keys():
        sd[k] = f.get_tensor(k)
dec_sd = {k.replace('decoder.',''): jt.array(v.astype(np.float32)) for k,v in sd.items() if k.startswith('decoder.')}
vae.decoder.load_state_dict(dec_sd)
lat = np.load('/data/PAM/ref/jt_final_unpacked.npy').astype(np.float32)
out = vae.decode(jt.array(lat)).numpy()
np.save('/data/PAM/ref/jt_decoded.npy', out)
img = (127.5*(np.clip(out[0].transpose(1,2,0),-1,1)+1)).astype(np.uint8)
cv2.imwrite('/data/PAM/ref/jt_final.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print('jt decoded', out.shape, 'min %.3f max %.3f' % (out.min(), out.max()), flush=True)
print('DONE')
