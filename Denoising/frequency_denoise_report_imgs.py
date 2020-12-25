import Space_Domain as sp
import cv2
import numpy as np
import tools.tools as T
import matplotlib.pyplot as plt

origin_path = "Imgs/lena.jpg"
noise = "gaussian"
img_path = "Imgs/Image_" + noise + ".jpg"
img_save_path = "Imgs/"
logfile_path = "Imgs/PSNR.txt"

plt.figure()

# 读取灰度图片
origin_img = cv2.imread(origin_path,0)
img = cv2.imread(img_path,0)
print("原始的峰值信噪比：",T.psnr(origin_img,img))
# Gaussian Filter 高斯滤波
out = sp.gauss.gaussian_filter(img, kernel_size=3, sigmax=0)
print("高斯滤波后的峰值信噪比：",T.psnr(origin_img,out))

ax1 = plt.subplot(1,2,1)
ax1.set_title("Gauss Filtering Result")
plt.imshow(out,cmap="gray")

# median filter 中值滤波
out = sp.median.MedianFilter(img_path,img_save_path + "Median_out.jpg",k=3,padding = None)
print("中值滤波后的峰值信噪比：",T.psnr(origin_img,out))

ax2 = plt.subplot(1,2,2)
ax2.set_title("Median Filtering Result")
plt.imshow(out,cmap="gray")
# plt.title(noise+" denoising results")
plt.show()
plt.figure()
# better median filter
out = sp.betterMedian.BetterMedianFilter(img_path,img_save_path + "BetterMedian_out.jpg",k=3,padding = None)
print("优化后中值滤波后的峰值信噪比：",T.psnr(origin_img,out))

ax1 = plt.subplot(1,2,1)
ax1.set_title("Optimized Median Filtering Result")
plt.imshow(out,cmap="gray")

# bilateral filter 双边滤波
out = cv2.bilateralFilter(src=img, d=0, sigmaColor=200, sigmaSpace=1)
# cv2.imwrite(img_save_path + "bilateral_out.jpg", out)
print("双边滤波后的峰值信噪比：",T.psnr(origin_img,out))

ax2 = plt.subplot(1,2,2)
ax2.set_title("Bilateral Filtering Result")
plt.imshow(out,cmap="gray")
# plt.title(noise+" denoising results")
plt.show()
plt.figure()
# NLmean 非局部均值滤波
sigma = 20
out = cv2.fastNlMeansDenoising(img, None, sigma, 5, 11)
# cv2.imwrite(img_save_path + "opencvNLmean_out.jpg", out)
print("opencv非局部均值滤波后的峰值信噪比：",T.psnr(origin_img,out))

ax1 = plt.subplot(1,2,1)
ax1.set_title("Opencv NLmean Filtering Result")
plt.imshow(out,cmap="gray")

out = sp.NLmean.NLmeansfilter(img.astype(np.float), sigma, 5, 11)
out = T.double2uint8(out)
# cv2.imwrite(img_save_path + "NLmean_out.jpg", out)
print("优化后非局部均值滤波后的峰值信噪比：",T.psnr(origin_img,out))

ax2 = plt.subplot(1,2,2)
ax2.set_title("NLmean Filtering Result")
plt.imshow(out,cmap="gray")
# plt.title(noise+" denoising results")
plt.show()