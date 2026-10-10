import numpy as np, cv2
a = np.load('/data/PAM/ref/pt_decoded.npy').astype(np.float64)   # torch
b = np.load('/data/PAM/ref/jt_decoded.npy').astype(np.float64)   # jittor
print('decoded tensor (range -1~1) MSE %.4e  MAX %.4e  RMSE %.4e'
      % ((a-b**2).mean() if False else ((a-b)**2).mean(), np.abs(a-b).max(), np.sqrt(((a-b)**2).mean())))
pa = cv2.imread('/data/PAM/ref/pt_final.png').astype(np.float64)
pb = cv2.imread('/data/PAM/ref/jt_final.png').astype(np.float64)
print('image (0-255) MSE %.4f  MAX %.1f  RMSE %.4f gray-levels'
      % (((pa-pb)**2).mean(), np.abs(pa-pb).max(), np.sqrt(((pa-pb)**2).mean())))
gap = np.full((pa.shape[0], 8, 3), 255)
cat = np.concatenate([pa, gap, pb], axis=1).astype(np.uint8)
cv2.imwrite('/data/PAM/ref/final_compare.png', cat)
diff = np.abs(pa-pb)
cv2.imwrite('/data/PAM/ref/final_diff_x10.png', np.clip(diff*10,0,255).astype(np.uint8))
cv2.imwrite('/data/PAM/ref/final_diff_x50.png', np.clip(diff*50,0,255).astype(np.uint8))
print('saved final_compare.png (左 torch / 右 jittor), final_diff_x10/x50.png')
print('DONE')
