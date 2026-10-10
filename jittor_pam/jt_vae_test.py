import sys, numpy as np, cv2, jittor as jt
sys.path.insert(0, '/data/PAM')
from jittor_pam.autoencoder import AutoEncoder, AutoEncoderParams
from safetensors import safe_open

params = AutoEncoderParams(resolution=1024, in_channels=3, ch=128, out_ch=3,
    ch_mult=[1,2,4,4], num_res_blocks=2, z_channels=16,
    scale_factor=0.3611, shift_factor=0.1159)
vae = AutoEncoder(params)
print('AE instantiated')
sd = {}
with safe_open('/data/PAM/weights/flux_dev/ae.safetensors', framework='np', device='cpu') as f:
    for k in f.keys():
        sd[k] = f.get_tensor(k)
print('ae.safetensors keys:', len(sd))
dec_sd = {k.replace('decoder.', ''): jt.array(v.astype(np.float32)) for k, v in sd.items() if k.startswith('decoder.')}
vae.decoder.load_state_dict(dec_sd)
w = vae.decoder.conv_in.weight
print('conv_in weight loaded:', np.allclose(w.numpy(), sd['decoder.conv_in.weight']))
lat = np.load('/data/PAM/ref/ref_latent.npy')
print('latent:', lat.shape, lat.dtype)
out = vae.decode(jt.array(lat))
out_np = out.numpy()
print('decoded:', out_np.shape, 'mean %.3f' % out_np.mean(), 'min %.3f max %.3f' % (out_np.min(), out_np.max()))
img = (127.5 * (np.clip(out_np[0].transpose(1, 2, 0), -1, 1) + 1)).astype(np.uint8)
cv2.imwrite('/data/PAM/ref/jt_ref.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
ref = cv2.imread('/data/PAM/ref/ref.png').astype(np.float32)
jt_ = cv2.imread('/data/PAM/ref/jt_ref.png').astype(np.float32)
print('MSE:', ((ref - jt_) ** 2).mean())
print('MAX abs diff:', np.abs(ref - jt_).max())
print('DONE')
