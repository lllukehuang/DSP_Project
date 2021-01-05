from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from skimage.io import imread
from scipy.fftpack import ifftn, fft2, ifft2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmath
from scipy import signal
import warnings
warnings.filterwarnings(action='ignore')


'''
对图像进行傅里叶变换，并获得频率信号的幅度 fimgamp、相位 fimgphase
'''


def fliproi(img, ranges):
    roi = img[ranges[0]:ranges[1], ranges[2]:ranges[3]]
    roi = np.flip(roi, 1)
    roi = np.flip(roi, 0)
    img[ranges[0]:ranges[1], ranges[2]:ranges[3]]=roi

def getShiftedFreqSpectrum(img):
    fimg = fft2(img, shape=img.shape, axes=tuple((0, 1)))
    fimgamp = np.abs(fimg)
    fimgphase = np.angle(fimg)
    roilen = img.shape[0]//2
    for y in [0, 1]:
        for x in [0, 1]:
            ranges = [y*roilen, (y+1)*roilen, x*roilen, (x+1)*roilen]
            fliproi(fimgamp, ranges)
            fliproi(fimgphase, ranges)
    return fimg, fimgamp, fimgphase

def getFreqSpectrum(img):
    fimg = fft2(img, shape=img.shape, axes=tuple((0, 1)))
    fimgamp = np.abs(fimg)
    fimgphase = np.angle(fimg)
    return fimg, fimgamp, fimgphase


img = cv2.imread('./imgs/gau.png', cv2.IMREAD_GRAYSCALE)
# 傅里叶变换频谱图、幅度谱、相位谱
fimg, fimgamp, fimgphase = getShiftedFreqSpectrum(img)
# 逆傅里叶变换
ishift = np.fft.ifftshift(np.fft.fftshift(fimg))
iimg = np.abs(np.fft.ifft2(ishift))

# plt.figure(figsize=(20,20))
# plt.subplot(1, 4, 1)
# plt.imshow(img, cmap="gray")
# plt.title("Orignal Image")
# plt.subplot(1, 4, 2)
# plt.imshow(np.log(fimgamp), cmap="gray")
# plt.title("Frequency Spectrum Amplification (Zero centered)")
# plt.subplot(1, 4, 3)
# plt.imshow(np.abs(fimgphase), cmap="gray")
# plt.title("Frequency Spectrum Phase Angle (Zero centered)")
# plt.subplot(1, 4, 4)
# plt.imshow(iimg, cmap='gray')
# plt.title("Inverse image")
# plt.show()


'''
构造高斯核
'''


def gkern(kernlen=22, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def gkern1(kernlen=22, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern2d = np.zeros((kernlen, kernlen), np.float32)
    gkern2d[kernlen//2, kernlen//2] = 1
    gkern2d = cv2.GaussianBlur(gkern2d, (kernlen, kernlen), std)
    return gkern2d


imgb = np.zeros((img.shape[0]+1, img.shape[1]+1), img.dtype)
if img.shape[0] % 2 == 0:
    imgb[0:img.shape[0], 0:img.shape[1]] = img
    fimg1, fimgamp1, fimgphase1 = getFreqSpectrum(imgb)

"""Gaussian Kernel"""
gkernel = gkern1(imgb.shape[0], 1)
fkernel, fkernelamp, fkernelphase = getFreqSpectrum(gkernel)

img1filtered1 = cv2.GaussianBlur(imgb, (imgb.shape[0], imgb.shape[1]), 1)
fimg1filtere2 = fkernel * fimg1
img1filtered2 = np.abs(ifft2(fimg1filtere2))
roilen = imgb.shape[0]//2
for y in [0, 1]:
        for x in [0, 1]:
            ranges = [y*roilen, (y+1)*roilen, x*roilen, (x+1)*roilen]
            fliproi(img1filtered2, ranges)
img1filtered2 = cv2.flip(img1filtered2, -1)
img1filtered2 = (img1filtered2 - img1filtered2.min())/(img1filtered2.max()-img1filtered2.min()) * 255
img1filtered1 = (img1filtered1 - img1filtered1.min())/(img1filtered1.max()-img1filtered1.min()) * 255

plt.figure(figsize=(15, 5))
plt.suptitle("Spatial Filter vs Frequency Filter", fontsize=14)
plt.subplot(1, 3, 1)
plt.imshow(img1filtered1, cmap="gray")
plt.title("Spatial Filter")
plt.subplot(1, 3, 2)
plt.imshow(img1filtered2, cmap="gray")
plt.title("Frequency Filter")
plt.subplot(1, 3, 3)
plt.imshow(np.abs(img1filtered2-img1filtered1), cmap="gray")
plt.title("The difference")
plt.show()
