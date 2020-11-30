import cv2
import noise

def main():
    imgpath = "test1.png"
    m = "random"
    cv2.imshow("Image", noise.noise(imgpath,"0"))
    cv2.imshow("Image" + m, noise.noise(imgpath, m))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()