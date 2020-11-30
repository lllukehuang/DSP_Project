import numpy as np

def psnr(A, B):
    return 10 * np.log(255 * 255.0 / (((A.astype(np.float) - B) ** 2).mean())) / np.log(10)


def double2uint8(I, ratio=1.0):
    return np.clip(np.round(I * ratio), 0, 255).astype(np.uint8)