import lpips
import torch
import glob
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

imgpaths = '/data/jzhang/github/aaai/3D-SGAN/output5000.0.jpg'
img = cv2.imread(imgpaths)
img0 = img[:,0:256,:]
img2 = img[:,-256:,:]

ssim = structural_similarity(img0, img2, gaussian_weights=True, sigma=1.5,
                             use_sample_covariance=False, multichannel=True,
                             data_range=img0.max() - img0.min())

print(ssim)

img0 = np.transpose(img0, axes=[2,0,1])[None,]
img0 = torch.from_numpy(img0)

img2 = np.transpose(img2, axes=[2,0,1])[None,]
img2 = torch.from_numpy(img2)
lpips = loss_fn_alex(img0, img2)

print(lpips)
