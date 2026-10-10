import sys
import importlib.util
import numpy as np
import torch
import jittor as jt

jt.flags.use_cuda = 1

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

pt_ae = load_module('pt_autoencoder', '/data/PAM/appearance_gen/src/flux/modules/autoencoder.py')
PTDecoder = pt_ae.Decoder

sys.path.insert(0, '/data/PAM')
from jittor_pam.autoencoder import Decoder as JTDecoder

kwargs = dict(ch=128, out_ch=3, ch_mult=[1,2,4,4], num_res_blocks=2,
              in_channels=3, resolution=256, z_channels=16)

pt_model = PTDecoder(**kwargs).cuda().eval()
jt_model = JTDecoder(**kwargs)

pt_state = pt_model.state_dict()
jt_state = {}
for k, v in pt_state.items():
    jt_state[k] = v.cpu().numpy()

jt_keys = set(jt_model.state_dict().keys())
pt_keys = set(pt_state.keys())
if jt_keys != pt_keys:
    print("=== KEY MISMATCH ===")
    print(f"Only in PyTorch: {sorted(pt_keys - jt_keys)}")
    print(f"Only in Jittor:   {sorted(jt_keys - pt_keys)}")
    print(f"PyTorch: {len(pt_keys)}, Jittor: {len(jt_keys)}")
else:
    print(f"All {len(pt_keys)} keys matched.")

jt_model.load_state_dict(jt_state)

np.random.seed(42)
np_input = np.random.randn(1, 16, 32, 32).astype(np.float32)
pt_input = torch.tensor(np_input).cuda()
jt_input = jt.array(np_input)

with torch.no_grad():
    pt_out = pt_model(pt_input).cpu().numpy()
jt_out = jt_model(jt_input).numpy()

diff = np.abs(pt_out - jt_out)
mse = np.mean((pt_out - jt_out) ** 2)
max_abs = np.max(diff)

print(f"\nPyTorch: shape={pt_out.shape}, range=[{pt_out.min():.4f},{pt_out.max():.4f}], mean={pt_out.mean():.4f}")
print(f"Jittor:  shape={jt_out.shape}, range=[{jt_out.min():.4f},{jt_out.max():.4f}], mean={jt_out.mean():.4f}")
print(f"MSE={mse:.2e}  MaxDiff={max_abs:.2e}")

if mse < 1e-5:
    print("\n*** NUMERICAL ALIGNMENT PASSED ***")
elif mse < 1e-3:
    print("\n*** MINOR DIFF (FP precision) ***")
else:
    print("\n*** FAIL: bug in port ***")
