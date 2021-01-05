import numpy as np
from scipy import misc
import pywt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

plt.figure(figsize=(15, 5))

image = misc.imread('./imgs/lena.png', mode='L')
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')

noiseSigma = 16.0
noised_image = image
noised_image += np.random.normal(0, noiseSigma, size=image.shape).astype(np.uint8)
noised_image = misc.imread('./imgs/gauss_out.jpg', mode='L')

wavelet = pywt.Wavelet('bior2.8')
levels = int(np.floor(np.log2(image.shape[0])))

WaveletCoeffs = pywt.wavedec2(noised_image, wavelet, level=levels)

threshold = noiseSigma * np.sqrt(2 * np.log2(image.size))
threshold = 20
NewWaveletCoeffs = list(map(lambda x: pywt.threshold(x, threshold), WaveletCoeffs))
NewImage = pywt.waverec2(NewWaveletCoeffs, wavelet)

plt.subplot(1, 3, 2)
plt.imshow(noised_image, cmap='gray')
plt.title('Adding Noise')
plt.subplot(1, 3, 3)
plt.imshow(NewImage, cmap='gray')
plt.title('New Image')
plt.show()


# def denoise(data, wavelet, noiseSigma):
#     levels = int(np.floor(np.log2(data.shape[0])))
#     WC = pywt.wavedec2(data, wavelet, level=levels)
#     threshold = noiseSigma * np.sqrt(2 * np.log2(data.size))
#     NWC = list(map(lambda x: pywt.threshold(x, threshold), WC))
#     return pywt.waverec2(NWC, wavelet)
#
# Denoised={}
# for wlt in pywt.wavelist():
#     Denoised[wlt] = denoise(data=image, wavelet=wlt, noiseSigma=16.0)
