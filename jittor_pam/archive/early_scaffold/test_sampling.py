import sys, math
import numpy as np
import torch
import jittor as jt

jt.flags.use_cuda = 1

sys.path.insert(0, '/data/PAM/appearance_gen/src')
from flux.sampling import (
    get_schedule as pt_get_schedule,
    unpack as pt_unpack,
    time_shift as pt_time_shift,
)

sys.path.insert(0, '/data/PAM')
from jittor_pam.sampling import (
    get_schedule as jt_get_schedule,
    unpack as jt_unpack,
    prepare as jt_prepare,
    get_noise as jt_get_noise,
    time_shift as jt_time_shift,
)

all_pass = True

# Test 1: get_schedule
print("=== Test 1: get_schedule ===")
for seq_len in [256, 1024, 4096]:
    pt_sched = pt_get_schedule(28, seq_len)
    jt_sched = jt_get_schedule(28, seq_len)
    mse = np.mean((np.array(pt_sched) - np.array(jt_sched)) ** 2)
    ok = mse < 1e-10
    print(f"  seq_len={seq_len}: MSE={mse:.2e} [{'PASS' if ok else 'FAIL'}]")
    all_pass &= ok

# Test 2: time_shift
print("\n=== Test 2: time_shift ===")
t = np.linspace(0.01, 1.0, 100)
pt_ts = pt_time_shift(0.7, 1.0, torch.tensor(t)).numpy()
jt_ts = jt_time_shift(0.7, 1.0, t)
mse = np.mean((pt_ts - jt_ts) ** 2)
ok = mse < 1e-10
print(f"  MSE={mse:.2e} [{'PASS' if ok else 'FAIL'}]")
all_pass &= ok

# Test 3: prepare (shape)
print("\n=== Test 3: prepare (shape) ===")
img = np.random.randn(1, 16, 64, 64).astype(np.float32)
txt = np.random.randn(1, 512, 4096).astype(np.float32)
vec = np.random.randn(1, 768).astype(np.float32)
inp = jt_prepare(jt.array(img), jt.array(txt), jt.array(vec))
print(f"  img: {inp['img'].shape} (expect [1, 1024, 64])")
print(f"  img_ids: {inp['img_ids'].shape} (expect [1, 1024, 3])")
print(f"  txt: {inp['txt'].shape}")
print(f"  txt_ids: {inp['txt_ids'].shape} (expect [1, 512, 3])")
print(f"  y: {inp['y'].shape}")
ok = inp['img'].shape == (1, 1024, 64) and inp['img_ids'].shape == (1, 1024, 3)
print(f"  [{'PASS' if ok else 'FAIL'}]")
all_pass &= ok

# Test 4: unpack (round-trip with prepare)
print("\n=== Test 4: unpack round-trip ===")
x = inp['img']  # [1, 1024, 64]
reconstructed = jt_unpack(x, height=512, width=512)
print(f"  unpacked: {reconstructed.shape} (expect [1, 16, 64, 64])")
ok = reconstructed.shape == (1, 16, 64, 64)
print(f"  [{'PASS' if ok else 'FAIL'}]")
all_pass &= ok

# Test 5: get_noise (shape + reproducibility)
print("\n=== Test 5: get_noise ===")
noise1 = jt_get_noise(1, 512, 512, seed=42)
noise2 = jt_get_noise(1, 512, 512, seed=42)
print(f"  shape: {noise1.shape} (expect [1, 16, 64, 64])")
repro = np.allclose(noise1.numpy(), noise2.numpy())
print(f"  reproducible: {repro}")
ok = noise1.shape == (1, 16, 64, 64) and repro
print(f"  [{'PASS' if ok else 'FAIL'}]")
all_pass &= ok

print("\n" + "=" * 40)
print("*** ALL SAMPLING TESTS PASSED ***" if all_pass else "*** SOME TESTS FAILED ***")
