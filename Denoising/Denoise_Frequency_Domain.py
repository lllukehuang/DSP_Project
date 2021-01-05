import cv2
import numpy as np
import tools.tools as T
import Frequency_Domain as fr 
import matplotlib.pyplot as plt

origin_path = "Imgs/lena_grey.jpg"
img_path = "Imgs/Image_gaussian.jpg"
img_save_path = "Imgs/"
logfile_path = "Imgs/PSNR2.txt"
f = open(logfile_path,"w",encoding='utf=8')

plt.figure()

# 读取灰度图片和加噪图片
origin_img = cv2.imread(origin_path,0)
img = cv2.imread(img_path,0)
print("原始的峰值信噪比：",T.psnr(origin_img,img))
f.write("原始的峰值信噪比："+str(T.psnr(origin_img,img)))
f.write("\n")

''' 
傅里叶变换
'''
out = fr.fourierFilter.fourierDenoise(img)
# Save result
cv2.imwrite(img_save_path + "./frequency/wavelet_out.jpg", out)
print("傅里叶变换滤波后的峰值信噪比：",T.psnr(origin_img, out))
f.write("傅里叶变换滤波后的峰值信噪比：" + str(T.psnr(origin_img,out)))
f.write("\n")

ax1 = plt.subplot(1,2,1)
ax1.set_title("Fourier transform Result")
plt.imshow(out, cmap="gray")


'''
小波变换
'''
out = fr.waveletFilter.w2d_test(img, mode='db1')
# Save result
cv2.imwrite(img_save_path + "./frequency/wavelet_out.jpg", out)
print("小波变换滤波后的峰值信噪比：",T.psnr(origin_img, out))
f.write("小波变换滤波后的峰值信噪比：" + str(T.psnr(origin_img,out)))
f.write("\n")

ax2 = plt.subplot(1,2,2)
ax2.set_title("Wavelet transform Result")
plt.imshow(out, cmap="gray")

plt.show()