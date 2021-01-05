import numpy as np
import matplotlib.pyplot as plt

import pywt

original = pywt.data.camera()

noiseSigma = 16.0
image = original + np.random.normal(0, noiseSigma, size=original.shape)

wavelet = pywt.Wavelet('haar')
levels  = int(np.floor(np.log2(image.shape[0])))

waveletCoeffs = pywt.wavedec2(image, wavelet, level=levels)

threshold = noiseSigma * np.sqrt(2 * np.log2(image.size)) 
newWaveletCoeffs = list(map(lambda x: pywt.threshold(x,threshold), waveletCoeffs))
newImage = pywt.waverec2(newWaveletCoeffs, wavelet)

plt.imshow(original)
plt.imshow(image)
plt.imshow(newImage)