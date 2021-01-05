import cv2
import noise

def main():
    imgpath = "lena.png"
    m = "s&p"
    cv2.imshow("Image", noise.noise(imgpath, "0"))
    cv2.imshow("Image" + m, noise.noise(imgpath, m))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
