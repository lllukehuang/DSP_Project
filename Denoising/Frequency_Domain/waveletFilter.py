import numpy as np
from scipy import misc
import pywt
import matplotlib.pyplot as plt
import warnings
import cv2
warnings.filterwarnings(action='ignore')

# noised_image = misc.imread(".\Frequency_Domain\imgs\lena.png",mode = "L")
# print(noised_image)
# print(type(noised_image))
# print(noised_image.dtype)
# print(noised_image.shape)
# exit(0)

def waveletDenoise(noised_image, threshold=20, Ty='bior2.8'):

    wavelet = pywt.Wavelet(Ty)
    levels = int(np.floor(np.log2(noised_image.shape[0])))

    WaveletCoeffs = pywt.wavedec2(noised_image, wavelet, level=levels)
    NewWaveletCoeffs = np.array(map(lambda x: pywt.threshold(x, threshold), WaveletCoeffs))
    NewImage = pywt.waverec2(NewWaveletCoeffs, wavelet)

    return NewImage

# waveletDenoise(noised_image)
# plt.subplot(1, 3, 2)
# plt.imshow(noised_image, cmap='gray')
# plt.title('Adding Noise')
# plt.subplot(1, 3, 3)
# plt.imshow(NewImage, cmap='gray')
# plt.title('New Image')
# plt.show()

def w2d(imArray, mode='haar'):
    #convert to float
    height,width = imArray.shape
    imArray =  np.float32(imArray)
    imArray /= 255
    # compute coefficients
    level = int(np.floor(np.log2(imArray.shape[0])))
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    threshold = 20
    coeffs = map(lambda x: pywt.threshold(x, threshold), coeffs)
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    # imArray_H = imArray_H[:223,:203]
    imArray_H.resize(height,width)

    return imArray_H




def w2d_test(imArray, mode='db1'):
    #convert to float
    # imArray = cv2.imread(img,0)
    height,width = imArray.shape
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    level = int(np.floor(np.log2(imArray.shape[0])))
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    threshold = 0

    # exit(0)
    # temp_list = coeffs.tolist()
    # print(temp_list)
    # coeffs_H = list(map(lambda x: pywt.threshold(x, threshold), coeffs))

    coeffs_H = list(coeffs)
    # print(coeffs_H)
    coeffs_H[0] *= 0
    # print(coeffs_H)

    # for i in range(len(coeffs_H)):
    #     print(type(coeffs_H[i]))

    # print("end")
    for i in range(1,len(coeffs_H)):
        temp_list = pywt.threshold(coeffs_H[i], threshold).tolist()
        coeffs_H[i] = tuple(temp_list)
        # coeffs_H[i] = pywt.threshold(coeffs_H[i], threshold)
    # print(coeffs_H)
    # for i in range(len(coeffs_H)):
    #     print(type(coeffs_H[i]))

    # coeffs_H = list(map(lambda x: pywt.threshold(x, threshold), coeffs_H))


    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    for i in range(height):
        for j in range(width):
            imArray_H[i][j] = 255 - imArray_H[i][j]
    # imArray_H = imArray_H[:223,:203]
    imArray_H.resize(height,width)
    cv2.imshow("wdnmd",imArray_H)
    cv2.waitKey(0)

    return imArray_H

# w2d_test("../Imgs/Image_gaussian.jpg")
# w2d_test("imgs/banma.jpg")
