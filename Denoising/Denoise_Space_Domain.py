import Space_Domain as sp
import cv2
import numpy as np
import tools.tools as T

origin_path = "Imgs/lena.jpg"
img_path = "Imgs/Image_s&p.jpg"
img_save_path = "Imgs/"
logfile_path = "Imgs/PSNR.txt"
f = open(logfile_path,"w",encoding='utf=8')
# 读取灰度图片和加噪图片
origin_img = cv2.imread(origin_path,0)
img = cv2.imread(img_path,0)
print("原始的峰值信噪比：",T.psnr(origin_img,img))
f.write("原始的峰值信噪比："+str(T.psnr(origin_img,img)))
f.write("\n")
# Gaussian Filter 高斯滤波
out = sp.gauss.gaussian_filter(img, kernel_size=3, sigmax=0)
# Save result
cv2.imwrite(img_save_path + "gauss_out.jpg", out)
# cv2.imshow("gauss_result", out)
print("高斯滤波后的峰值信噪比：",T.psnr(origin_img,out))
f.write("高斯滤波后的峰值信噪比："+str(T.psnr(origin_img,out)))
f.write("\n")
# median filter 中值滤波
out = sp.median.MedianFilter(img_path,img_save_path + "Median_out.jpg",k=3,padding = None)
# cv2.imwrite(img_save_path + "Median_out.jpg", out)
out.save(img_save_path + "Median_out.jpg")
print("中值滤波后的峰值信噪比：",T.psnr(origin_img,out))
f.write("中值滤波后的峰值信噪比："+str(T.psnr(origin_img,out)))
f.write("\n")
# better median filter
out = sp.betterMedian.BetterMedianFilter(img_path,img_save_path + "BetterMedian_out.jpg",k=3,padding = None)
# cv2.imwrite(img_save_path + "BetterMedian_out.jpg",out)
out.save(img_save_path + "BetterMedian_out.jpg")
print("优化后中值滤波后的峰值信噪比：",T.psnr(origin_img,out))
f.write("优化后中值滤波后的峰值信噪比："+str(T.psnr(origin_img,out)))
f.write("\n")
# bilateral filter 双边滤波
out = cv2.bilateralFilter(src=img, d=0, sigmaColor=100, sigmaSpace=1)
cv2.imwrite(img_save_path + "bilateral_out.jpg", out)
print("双边滤波后的峰值信噪比：",T.psnr(origin_img,out))
f.write("双边滤波后的峰值信噪比："+str(T.psnr(origin_img,out)))
f.write("\n")
# NLmean 非局部均值滤波
sigma = 20
out = cv2.fastNlMeansDenoising(img, None, sigma, 5, 11)
cv2.imwrite(img_save_path + "opencvNLmean_out.jpg", out)
print("opencv非局部均值滤波后的峰值信噪比：",T.psnr(origin_img,out))
f.write("opencv非局部均值滤波后的峰值信噪比："+str(T.psnr(origin_img,out)))
f.write("\n")
out = sp.NLmean.NLmeansfilter(img.astype(np.float), sigma, 5, 11)
out = T.double2uint8(out)
cv2.imwrite(img_save_path + "NLmean_out.jpg", out)
print("优化后非局部均值滤波后的峰值信噪比：",T.psnr(origin_img,out))
f.write("优化后非局部均值滤波后的峰值信噪比："+str(T.psnr(origin_img,out)))
f.write("\n")
f.close()
print("图片已保存至"+img_save_path+"文件夹下")