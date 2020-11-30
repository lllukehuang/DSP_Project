# 双边滤波
# 其和高斯滤波的主要区别是增加了值域核
# 最终权重由函数核与值域核相乘得到

import cv2
def bi_demo(image):#高斯双边滤波
    dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=15)
    cv2.namedWindow('bi_demo',0)
    cv2.resizeWindow('bi_demo',300,400)
    cv2.imshow("bi_demo", dst)

'''
    其中各参数所表达的意义：
    src：原图像；
    d：像素的邻域直径，可有sigmaColor和sigmaSpace计算可得；
    sigmaColor：颜色空间的标准方差，一般尽可能大；
    sigmaSpace：坐标空间的标准方差(像素单位)，一般尽可能小。'''

def mean_shift_demo(image):#均值偏移滤波
    dst = cv2.pyrMeanShiftFiltering(src=image, sp=15, sr=20)
    cv2.namedWindow('mean_shift image', 0)
    cv2.resizeWindow('mean_shift image', 300, 400)
    cv2.imshow("mean_shift image", dst)


#使用均值边缘保留滤波时，可能会导致图像过度模糊
'''其中各参数所表达的意义：
    src：原图像;
    sp：空间窗的半径(The spatial window radius);
    sr：色彩窗的半径(The color window radius)'''

if __name__ == '__main__':
    src = cv2.imread('lena.jpg')
    bi_demo(src)
    mean_shift_demo(src)
    cv2.namedWindow('src', 0)
    cv2.resizeWindow('src', 300, 400)
    cv2.imshow('src',src)
    cv2.waitKey(0)
