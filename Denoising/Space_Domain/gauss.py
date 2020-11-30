# 高斯滤波
import cv2
import numpy as np

def gaussian_filter3d(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1) # 把灰度图扩充一个维度
        H, W, C = img.shape
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    ## prepare Kernel 生成高斯模板
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    tmp = out.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c]) # 卷积
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out


def gaussian_filter(img, kernel_size, sigmax):
    row, col = img.shape  # 获得未添加边界前的大小信息

    # 下面产生卷积核
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2  # 整除

    # 计算标准差
    if sigmax == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    else:
        sigma = sigmax

    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center  # center-j也无所谓，反正权重是按到圆心距离算的，而且距离带平方了，正负无所谓，x**2+y**2的值代表了权重。
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
            # /(np.pi * s)
    sum_val = 1 / sum_val
    # 对卷积核归一化，确保卷积核之和为1
    kernel = kernel * sum_val  # 对于np.array来说是对应位置相乘。这里不要用除，最好是用乘以分之一的形式
    # 以上是产生卷积核

    # 计算图像边界需要添加的范围，扩充图像边界使得能遍历到原图像的每一个像素
    addLine = int((kernel_size - 1) / 2)  # 虽然addLine理应是整数，但存储类型是浮点，要转换类型
    img = cv2.copyMakeBorder(img, addLine, addLine, addLine, addLine, borderType=cv2.BORDER_REPLICATE)

    # 定位未扩充之前图像左上角元素在新图像中的下标索引,这个索引将用来遍历原图像的每一个像素点，相当于指针
    source_x = addLine  # 定义起始位置，即未扩充之前图像左上角元素在新图像中的下标索引
    source_y = addLine  # 定义起始位置，即未扩充之前图像左上角元素在新图像中的下标索引
    # addLine的值是单边添加边界的大小（行数，也是列数），一共四个边添加边界

    # 在添加了边界后的图像中遍历未添加边界时的原像素点，进行滤波
    # 外层控制行，内层控制列
    for delta_x in range(0, row):
        for delta_y in range(0, col):
            img[source_x, source_y] = np.sum(
                img[source_x - addLine:source_x + addLine + 1, source_y - addLine:source_y + addLine + 1] * kernel)
            source_y = source_y + 1
        source_x = source_x + 1  # 行加1，准备进行下行的所有列的遍历
        source_y = addLine  # 把列归位到原始的列起点准备下轮列遍历
    # 经过上面的循环后，图像已经滤波完成了

    # 剥除四边添加的边界，然后返回滤波后的图片
    return img[addLine:row + addLine, addLine:col + addLine]


# Read image
if __name__ == '__main__':
    img = cv2.imread("lena.jpg")
    # Gaussian Filter
    out = gaussian_filter3d(img, K_size=3, sigma=1.3)
    # Save result
    cv2.imwrite("out.jpg", out)
    cv2.imshow("result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
