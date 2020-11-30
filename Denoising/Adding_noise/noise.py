import cv2
import skimage
import numpy as np
import random

'''
m is a str type parameter, which decide the type of noise to add;
In our project, we only used gaussian, salt and pepper noise.
‘0’ No noise.
‘gaussian’ Gaussian-distributed additive noise.
‘localvar’ Gaussian-distributed additive noise, with specified local variance at each point of image
‘poisson’ Poisson-distributed noise generated from the data.
‘salt’ Replaces random pixels with 1.
‘pepper’ Replaces random pixels with 0 (for unsigned images) or -1 (for signed images).
‘s&p’ Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signedimages.
‘speckle’ Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
'''

def sknoise( imgpath , m ) :
    # use scikit-image pack
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if m == '0':
        return img

    img = skimage.util.random_noise(img, mode=m, seed=None, clip=True)
    return img

def noise( imgpath , m ) :

    img = cv2.imread(imgpath)
    #print(img.shape)
    noise_num = 3000 # the number of random noise
    prob = 0.1 # the prob of pepper noise in s&p
    thres = 1 - prob # the prob of salt noise in s&p
    mean = 0 # mean in gaussian noise
    var = 0.001 # variance in gaussian noise

    if m == "random" :
        rows, cols, chn = img.shape
        for i in range(noise_num):
            x = np.random.randint(0, rows)  # 随机生成指定范围的整数
            y = np.random.randint(0, cols)
            img[x, y,:] = 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    if m == "s&p":
        output = np.zeros(img.shape, np.uint8)
        noise_out = np.zeros(img.shape, np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                    noise_out[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                    noise_out[i][j] = 255
                else:
                    output[i][j] = img[i][j]
                    noise_out[i][j] = 100
        img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        #print(img.shape)
        return img

    if m == "pepper":
        output = np.zeros(img.shape, np.uint8)
        noise_out = np.zeros(img.shape, np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                    noise_out[i][j] = 0
                else:
                    output[i][j] = img[i][j]
                    noise_out[i][j] = 100
        img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        return img

    if m == "salt":
        output = np.zeros(img.shape, np.uint8)
        noise_out = np.zeros(img.shape, np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn > thres:
                    output[i][j] = 255
                    noise_out[i][j] = 255
                else:
                    output[i][j] = img[i][j]
                    noise_out[i][j] = 100
        img = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        return img

    if m == "gaussian" :
        img = np.array(img / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, img.shape)
        out = img + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        img = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        return img

    else :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img