import cv2
import numpy as np
import tools.tools as T
import Frequency_Domain as fr

origin_path = "Imgs/lena.jpg"
img_path = "Imgs/Image_s&p.jpg"
img_save_path = "Imgs/"
logfile_path = "Imgs/PSNR2.txt"
f = open(logfile_path,"w",encoding='utf=8'