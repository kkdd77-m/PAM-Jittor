import sys
sys.path.insert(0, '/data/PAM')
import jittor as jt
from jittor_pam.autoencoder import Decoder, Encoder, AutoEncoder, AutoEncoderParams

jt.flags.use_cuda = 1

params = AutoEncoderParams(
    resolution=256, in_channels=3, ch=128, out_ch=3,
    ch_mult=[1, 2, 4, 4], num_res_blocks=2, z_channels=16,
    scale_factor=0.3611, shift_factor=0.1159,
)

print("=== Test 1: Decoder shape ===")
decoder = Decoder(ch=128, out_ch=3, ch_mult=[1,2,4,4], num_res_blocks=2,
                  in_channels=3, resolution=256, z_channels=16)
z = jt.randn((1, 16, 32, 32))
out = decoder(z)
print(f"  input:  {z.shape}")
print(f"  output: {out.shape}")
assert out.shape == (1, 3, 256, 256), f"FAIL: {out.shape}"
print("  PASSED")

print("=== Test 2: Encoder shape ===")
encoder = Encoder(resolution=256, in_channels=3, ch=128,
                  ch_mult=[1,2,4,4], num_res_blocks=2, z_channels=16)
img = jt.randn((1, 3, 256, 256))
enc_out = encoder(img)
print(f"  input:  {img.shape}")
print(f"  output: {enc_out.shape}")
assert enc_out.shape == (1, 32, 32, 32), f"FAIL: {enc_out.shape}"
print("  PASSED")

print("=== Test 3: Full AutoEncoder round-trip ===")
ae = AutoEncoder(params)
recon = ae(img)
print(f"  input:  {img.shape}")
print(f"  output: {recon.shape}")
assert recon.shape == img.shape, f"FAIL: {recon.shape}"
print("  PASSED")

print("\n=== ALL VAE SHAPE TESTS PASSED ===")
