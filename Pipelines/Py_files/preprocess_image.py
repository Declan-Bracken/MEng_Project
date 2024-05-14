import cv2
import numpy as np

class PreprocessImg():
    def __init__(self, img_path):
        image = cv2.imread(img_path)

        # deskewed_image = self.deskew(image)
        gray = self.get_grayscale(image)
        thresh = self.thresholding(gray)
        cv2.imshow('thresh',thresh)
        cv2.waitKey(0)
        opening = self.opening(gray)
        cv2.imshow('opening',opening)
        cv2.waitKey(0)
        canny = self.canny(gray)
        cv2.imshow('canny',canny)
        cv2.waitKey(0)
        # and finally destroy/close all open windows
        cv2.destroyAllWindows()

    # get grayscale image
    def get_grayscale(self,image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self,image):
        return cv2.medianBlur(image,5)
    
    #thresholding
    def thresholding(self,image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #dilation
    def dilate(self,image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)
        
    #erosion
    def erode(self,image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)

    #opening - erosion followed by dilation
    def opening(self,image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    #canny edge detection
    def canny(self,image):
        return cv2.Canny(image, 100, 200)

    #skew correction
    def deskew(self,image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

if __name__ == "__main__":
    image_path = r'\Users\Declan Bracken\Pictures\Saved Pictures\2015-queens-university-transcript-1-2048.webp'
    PreprocessImg(image_path)