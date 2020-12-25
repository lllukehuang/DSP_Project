# 中值滤波1
from PIL import Image
import numpy as np

def MedianFilter(src, dst, k=3, padding=None):
    imarray = np.array(Image.open(src).convert('L'))
    height, width = imarray.shape

    if not padding:
        edge = int((k - 1) / 2)
        if height - 1 - edge <= edge or width - 1 - edge <= edge:
            print("The parameter k is too large.")
            return None
        new_arr = np.zeros((height, width), dtype="uint8")
        for i in range(height):
            for j in range(width):
                if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= height - edge - 1:
                    new_arr[i, j] = imarray[i, j]
                else:
                    new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1]) # 计算中位数
        new_im = Image.fromarray(new_arr)
        # new_im.save(dst)
    return new_im

if __name__ == '__main__':
    src = "lena.jpg" # source image path
    dst = "out.jpg"

    MedianFilter(src, dst)