import cv2
import Adding_noise.noise as noise

def main():
    imgpath = "Imgs/lena.jpg"
    m = "pepper"
    cv2.imshow("Image", noise.noise(imgpath,"0"))
    cv2.imshow("Image_" + m, noise.noise(imgpath, m))
    cv2.imwrite("Imgs/Image_" + m + ".jpg", noise.noise(imgpath, m))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()