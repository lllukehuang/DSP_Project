# 中值滤波2
from PIL import Image
import numpy as np


def BetterMedianFilter(src, dst, k=3, padding=None):
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
                    # nm:neighbour matrix
                    # 只对相邻滤波器窗口内的最大值或最小值进行中值滤波，避免不必要的清晰度损失
                    nm = imarray[i - edge:i + edge + 1, j - edge:j + edge + 1]
                    max = np.max(nm)
                    min = np.min(nm)
                    if imarray[i, j] == max or imarray[i, j] == min:
                        new_arr[i, j] = np.median(nm)
                    else:
                        new_arr[i, j] = imarray[i, j]
        new_im = Image.fromarray(new_arr)
        new_im.save(dst)
    return new_im

if __name__ == '__main__':
    src = "lena.jpg"
    dst = "out.jpg"

    BetterMedianFilter(src, dst)