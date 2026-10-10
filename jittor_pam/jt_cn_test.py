import sys, numpy as np, jittor as jt
sys.path.insert(0, '/data/PAM')
from jittor_pam.controlnet import ControlNetFlux, FluxParams
params = FluxParams(in_channels=64, vec_in_dim=768, context_in_dim=4096,
    hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19,
    depth_single_blocks=38, axes_dim=[16,56,56], theta=10_000,
    qkv_bias=True, guidance_embed=True)
sd = np.load('/data/PAM/ref/controlnet_fp32.npz')
def load_cn(prefix):
    cn_sd = {k[len(prefix):]: jt.array(v) for k, v in sd.items() if k.startswith(prefix)}
    cn = ControlNetFlux(params, controlnet_depth=2, condition_in_channels=6)
    cn.load_state_dict(cn_sd)
    return cn
cn1 = load_cn('controlnet1.')
cn2 = load_cn('controlnet2.')
print('CN loaded (jittor)')
conds = np.load('/data/PAM/ref/conds.npz')
depth = jt.array(conds['depth']); normal = jt.array(conds['normal'])
hand = jt.array(conds['hand']); seg = jt.array(conds['seg'])
physical = jt.concat([depth, normal], dim=1)
semantic = jt.concat([hand, seg], dim=1)
inp = {k: jt.array(v) for k, v in np.load('/data/PAM/ref/inp_true.npz').items()}
t_curr = float(np.load('/data/PAM/ref/t_curr.npy'))
t_vec = jt.full((1,), t_curr, dtype=jt.float32)
guidance_vec = jt.full((1,), 4.0, dtype=jt.float32)
print('inputs ready', flush=True)
r1 = cn1(**inp, controlnet_cond=physical, timesteps=t_vec, guidance=guidance_vec)
r2 = cn2(**inp, controlnet_cond=semantic, timesteps=t_vec, guidance=guidance_vec)
ref = np.load('/data/PAM/ref/cn_out.npz')
for name, r in [('cn1_0', r1[0]), ('cn1_1', r1[1]), ('cn2_0', r2[0]), ('cn2_1', r2[1])]:
    a = ref[name].astype(np.float64)
    b = r.numpy().astype(np.float64)
    print(name, 'MSE %.4e  MAX %.4e' % (((a-b)**2).mean(), np.abs(a-b).max()))
print('DONE')
