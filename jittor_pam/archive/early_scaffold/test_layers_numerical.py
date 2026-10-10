import sys
import numpy as np
import torch
import jittor as jt

jt.flags.use_cuda = 1

# ========== 加载 PyTorch 原版 ==========
sys.path.insert(0, '/data/PAM/appearance_gen/src')
from flux.math import attention as pt_attention, rope as pt_rope
from flux.modules.layers import (
    timestep_embedding as pt_timestep_embedding,
    EmbedND as PTEmbedND,
    RMSNorm as PTRMSNorm,
    Modulation as PTModulation,
    DoubleStreamBlock as PTDoubleStreamBlock,
    SingleStreamBlock as PTSingleStreamBlock,
    LastLayer as PTLastLayer,
)

# ========== 加载 Jittor 版 ==========
sys.path.insert(0, '/data/PAM')
from jittor_pam.layers import (
    timestep_embedding as jt_timestep_embedding,
    rope as jt_rope,
    attention as jt_attention,
    EmbedND as JTEmbedND,
    RMSNorm as JTRMSNorm,
    Modulation as JTModulation,
    DoubleStreamBlock as JTDoubleStreamBlock,
    SingleStreamBlock as JTSingleStreamBlock,
    LastLayer as JTLastLayer,
)

print("Import OK")

# ========== 工具函数 ==========
def copy_weights(pt_model, jt_model, name=""):
    pt_state = pt_model.state_dict()
    jt_state = {k: v.cpu().numpy() for k, v in pt_state.items()}
    jt_keys = set(jt_model.state_dict().keys())
    pt_keys = set(pt_state.keys())
    if jt_keys != pt_keys:
        print(f"  [KEY MISMATCH] {name}")
        print(f"    PT-only: {sorted(pt_keys - jt_keys)[:5]}")
        print(f"    JT-only: {sorted(jt_keys - pt_keys)[:5]}")
        return False
    jt_model.load_state_dict(jt_state)
    return True

def compare(name, pt_out, jt_out, tol=1e-4):
    pt_np = pt_out.detach().cpu().numpy() if isinstance(pt_out, torch.Tensor) else np.array(pt_out)
    jt_np = jt_out.numpy() if isinstance(jt_out, jt.Var) else np.array(jt_out)
    mse = np.mean((pt_np - jt_np) ** 2)
    maxdiff = np.max(np.abs(pt_np - jt_np))
    ok = mse < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: MSE={mse:.2e} MaxDiff={maxdiff:.2e}")
    return ok

def make_ids(B, L):
    h = int(np.ceil(np.sqrt(L)))
    ids = np.stack(np.meshgrid(np.arange(h), np.arange(h), indexing='ij'), -1)
    ids = ids.reshape(1, -1, 2)[:, :L, :].astype(np.float32)
    return np.broadcast_to(ids, (B, L, 2)).copy()

np.random.seed(42)
all_pass = True

# ========== Test 1: timestep_embedding ==========
print("\n=== Test 1: timestep_embedding ===")
t = np.random.rand(4).astype(np.float32) * 500
pt_out = pt_timestep_embedding(torch.tensor(t).cuda(), dim=256)
jt_out = jt_timestep_embedding(jt.array(t), dim=256)
all_pass &= compare("timestep_emb", pt_out, jt_out, 1e-5)

# ========== Test 2: rope ==========
print("\n=== Test 2: rope ===")
pos = np.random.randint(0, 10, (2, 16)).astype(np.float32)
pt_out = pt_rope(torch.tensor(pos).cuda(), dim=64, theta=10000)
jt_out = jt_rope(jt.array(pos), dim=64, theta=10000)
all_pass &= compare("rope", pt_out, jt_out, 1e-5)

# ========== Test 3: EmbedND ==========
print("\n=== Test 3: EmbedND ===")
pt_emb = PTEmbedND(dim=64, theta=10000, axes_dim=[32, 32]).cuda()
jt_emb = JTEmbedND(dim=64, theta=10000, axes_dim=[32, 32])
ids = make_ids(1, 16)
pt_pe = pt_emb(torch.tensor(ids).cuda())
jt_pe = jt_emb(jt.array(ids))
all_pass &= compare("EmbedND", pt_pe, jt_pe, 1e-5)

# ========== Test 4: attention + RoPE ==========
print("\n=== Test 4: attention (with RoPE) ===")
B, H, L, D = 1, 4, 16, 64
q = np.random.randn(B, H, L, D).astype(np.float32)
k = np.random.randn(B, H, L, D).astype(np.float32)
v = np.random.randn(B, H, L, D).astype(np.float32)
pt_out = pt_attention(torch.tensor(q).cuda(), torch.tensor(k).cuda(),
                       torch.tensor(v).cuda(), pe=pt_pe)
jt_out = jt_attention(jt.array(q), jt.array(k), jt.array(v), pe=jt_pe)
all_pass &= compare("attention", pt_out, jt_out, 1e-3)

# ========== Test 5: RMSNorm ==========
print("\n=== Test 5: RMSNorm ===")
pt_m = PTRMSNorm(64).cuda().eval()
jt_m = JTRMSNorm(64)
copy_weights(pt_m, jt_m, "RMSNorm")
x = np.random.randn(2, 10, 64).astype(np.float32)
all_pass &= compare("RMSNorm", pt_m(torch.tensor(x).cuda()), jt_m(jt.array(x)), 1e-5)

# ========== Test 6: Modulation ==========
print("\n=== Test 6: Modulation (double) ===")
pt_m = PTModulation(64, double=True).cuda().eval()
jt_m = JTModulation(64, double=True)
if copy_weights(pt_m, jt_m, "Modulation"):
    vec = np.random.randn(2, 64).astype(np.float32)
    pt_o1, pt_o2 = pt_m(torch.tensor(vec).cuda())
    jt_o1, jt_o2 = jt_m(jt.array(vec))
    all_pass &= compare("mod shift1", pt_o1.shift, jt_o1[0], 1e-5)
    all_pass &= compare("mod scale1", pt_o1.scale, jt_o1[1], 1e-5)
    all_pass &= compare("mod gate1",  pt_o1.gate,  jt_o1[2], 1e-5)
    all_pass &= compare("mod shift2", pt_o2.shift, jt_o2[0], 1e-5)

# ========== Test 7: DoubleStreamBlock ==========
print("\n=== Test 7: DoubleStreamBlock (END-TO-END) ===")
HS, NH, MR = 64, 4, 4.0
head_dim = HS // NH  # 16
pt_blk = PTDoubleStreamBlock(HS, NH, MR).cuda().eval()
jt_blk = JTDoubleStreamBlock(HS, NH, MR)
if copy_weights(pt_blk, jt_blk, "DoubleStreamBlock"):
    B, L_img, L_txt = 1, 16, 8
    L_tot = L_img + L_txt
    img = np.random.randn(B, L_img, HS).astype(np.float32)
    txt = np.random.randn(B, L_txt, HS).astype(np.float32)
    vec = np.random.randn(B, HS).astype(np.float32)
    # pe 需要 head_dim=16, L=L_tot
    pt_emb2 = PTEmbedND(dim=head_dim, theta=10000, axes_dim=[head_dim//2, head_dim//2]).cuda()
    jt_emb2 = JTEmbedND(dim=head_dim, theta=10000, axes_dim=[head_dim//2, head_dim//2])
    ids2 = make_ids(B, L_tot)
    pe_pt = pt_emb2(torch.tensor(ids2).cuda())
    pe_jt = jt_emb2(jt.array(ids2))

    with torch.no_grad():
        pt_img_o, pt_txt_o = pt_blk(torch.tensor(img).cuda(), torch.tensor(txt).cuda(),
                                     torch.tensor(vec).cuda(), pe_pt)
    jt_img_o, jt_txt_o = jt_blk(jt.array(img), jt.array(txt), jt.array(vec), pe_jt)
    all_pass &= compare("DSB img", pt_img_o, jt_img_o, 1e-3)
    all_pass &= compare("DSB txt", pt_txt_o, jt_txt_o, 1e-3)

# ========== Test 8: SingleStreamBlock ==========
print("\n=== Test 8: SingleStreamBlock (END-TO-END) ===")
pt_blk = PTSingleStreamBlock(HS, NH, MR).cuda().eval()
jt_blk = JTSingleStreamBlock(HS, NH, MR)
if copy_weights(pt_blk, jt_blk, "SingleStreamBlock"):
    x = np.random.randn(1, 24, HS).astype(np.float32)
    vec = np.random.randn(1, HS).astype(np.float32)
    ids3 = make_ids(1, 24)
    pe_pt = pt_emb2(torch.tensor(ids3).cuda())
    pe_jt = jt_emb2(jt.array(ids3))
    with torch.no_grad():
        pt_o = pt_blk(torch.tensor(x).cuda(), torch.tensor(vec).cuda(), pe_pt)
    jt_o = jt_blk(jt.array(x), jt.array(vec), pe_jt)
    all_pass &= compare("SSB", pt_o, jt_o, 1e-3)

# ========== Test 9: LastLayer ==========
print("\n=== Test 9: LastLayer ===")
pt_m = PTLastLayer(HS, patch_size=2, out_channels=16).cuda().eval()
jt_m = JTLastLayer(HS, patch_size=2, out_channels=16)
if copy_weights(pt_m, jt_m, "LastLayer"):
    x = np.random.randn(1, 16, HS).astype(np.float32)
    vec = np.random.randn(1, HS).astype(np.float32)
    with torch.no_grad():
        pt_o = pt_m(torch.tensor(x).cuda(), torch.tensor(vec).cuda())
    jt_o = jt_m(jt.array(x), jt.array(vec))
    all_pass &= compare("LastLayer", pt_o, jt_o, 1e-4)

# ========== Summary ==========
print("\n" + "=" * 50)
print("*** ALL LAYERS PASSED ***" if all_pass else "*** SOME TESTS FAILED ***")
