# 在 torch 环境运行：/opt/torchvenv/bin/python export_cn_npz.py
# 从 controlnet.bin 再生 Jittor 用的 CN 权重（推理用 fp16；训练梯度对齐用 bf16u16）
import torch, numpy as np, os
src='/data/PAM/weights/pam_ckpt/appearance_gen/controlnet.bin'
sd=torch.load(src,map_location='cpu',weights_only=True)
os.makedirs('/data/PAM/ref',exist_ok=True)
np.savez('/data/PAM/ref/controlnet_fp16.npz', **{k:v.detach().half().numpy() for k,v in sd.items()})
np.savez('/data/PAM/ref/controlnet_bf16u16.npz', **{k:v.detach().view(torch.uint16).numpy().astype('<u2') for k,v in sd.items()})
print('regenerated controlnet_fp16.npz + controlnet_bf16u16.npz, tensors=%d'%len(sd))
