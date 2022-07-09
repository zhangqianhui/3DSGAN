import lpips
import torch
import glob
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

imgpaths = '/data/jzhang/github/aaai/3D-SGAN/lpips'
imglists = list(glob.glob(os.path.join(imgpaths, '*.jpg')))

img0 = cv2.imread(imglists[0])
img0np = img0.copy()
img0 = np.transpose(img0, axes=[2,0,1])[None,]
img0 = torch.from_numpy(img0)

avg_lpips = 0.0
avg_ssim = 0.0
for i in range(len(imglists)):
    img1 = cv2.imread(imglists[i])
    ssim = structural_similarity(img1, img0np, gaussian_weights=True, sigma=1.5,
                        use_sample_covariance=False, multichannel=True,
                        data_range=img0np.max() - img0np.min())
    img1 = np.transpose(img1, axes=[2, 0, 1])[None,]
    img1 = torch.from_numpy(img1)
    d = loss_fn_alex(img0, img1)
    avg_lpips += d
    avg_ssim += ssim

print('lpips', avg_lpips/len(imglists), 'ssim', avg_ssim / len(imglists))
#