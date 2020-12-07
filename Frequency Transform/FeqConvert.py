import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt,pi


def forwardfft(img_grey):
    img_freq = np.fft.fft2(img_grey)
    img_freq_center = np.fft.fftshift(img_freq)
    return img_freq_center


def ifft(freq):
    fft = np.fft.ifftshift(freq)
    img = np.fft.ifft2(fft)
    img = np.real(img)

    max = np.max(img)
    min = np.min(img)

    h,w = img.shape

    for i in range(h):
        for j in range(w):
            img[i, j] = 255 * (img[i, j] - min)/(max - min)
    return img


def GaussianHighPFreqFilter(freq, d):
    h, w = freq.shape
    center_point = (h / 2, w / 2)
    for i in range(h):
        for j in range(w):
            dis = sqrt((i - center_point[0]) ** 2 + (j - center_point[1]) ** 2)
            freq[i, j] *= (1 - np.exp(-(dis ** 2) / (2 * (d ** 2))))
    return freq


def GaussianLowPFreqFilter(freq, d):
    h, w = freq.shape
    center_point = (h / 2, w / 2)
    for i in range(h):
        for j in range(w):
            dis = sqrt((i - center_point[0]) ** 2 + (j - center_point[1]) ** 2)
            freq[i, j] *= (np.exp(-(dis ** 2) / (2 * (d ** 2))))
    return freq


def LaplaceShapen(freq):
    h, w = freq.shape
    center_point = (h / 2, w / 2)
    for i in range(h):
        for j in range(w):
            freq[i, j] *= -4*pi*pi*((i - center_point[0]) ** 2 + (j - center_point[1]) ** 2)
    img_out = ifft(freq)
    print(img_out)
    img_out = np.real(img_out)
    max = np.max(img_out)
    min = np.min(img_out)

    res = np.zeros((h, w), dtype="uint8")

    for i in range(h):
        for j in range(w):
            res[i, j] = 255 * (img_out[i, j] - min) / (max - min)
    return img_out


img = cv2.imread("1.jpg")
### change the channel from BGR to RGB####
b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])
#### change to grey scale####
img_grey = r * 0.299 + g * 0.587 + b * 0.114

cv2.imwrite("grey_scale.png",img_grey)

freq1 = forwardfft(img_grey)
freq2 = freq1.copy()
freq3 = freq1.copy()
freq4 = freq1.copy()
res = np.log(np.abs(freq1))


freq1 = GaussianLowPFreqFilter(freq1, 40)
freq2 = GaussianHighPFreqFilter(freq2, 40)
freq3 = LaplaceShapen(freq3)

img2 = ifft(freq1)
img3 = ifft(freq2)
#img4 = ifft(freq3)

sharpen_img = img_grey-freq3

max = np.max(img_grey)
min = np.min(img_grey)

h,w = sharpen_img.shape

for i in range(h):
    for j in range(w):
        sharpen_img[i, j] = 255*(sharpen_img[i, j] - min)/(max - min)


plt.imshow(sharpen_img, cmap='gray')
plt.axis('off')
#plt.show()
cv2.imwrite("gauss_lowp.png", img2)
cv2.imwrite("gauss_highp.png", img3)
print(sharpen_img)
cv2.imwrite("Lap.png",sharpen_img)
'''
img_out = cv2.Laplacian(img_grey, ddepth=cv2.CV_64F)
img_out = cv2.convertScaleAbs(img_grey-img_out)
cv2.imshow("1",img_out)
cv2.waitKey()
'''
