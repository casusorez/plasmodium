from scipy import ndimage
from skimage.morphology import watershed
import cv2
import numpy as np
from skimage.feature import peak_local_max

from skimage import color
from skimage import io
import matplotlib.pyplot as plt

def detect(img) :
    plt.imshow(img)
    plt.title("Source")
    plt.show()
    
    gray = np.uint8(color.rgb2gray(img) * 255)
    plt.imshow(gray, cmap=plt.get_cmap("gray"))
    plt.title("Gray Scale")
    plt.show()
    
    #gray c'est ton image en niveaux de gris
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 3) #3 ou 2 max ********************************
    thresh = cv2.dilate(erosion,kernel,iterations = 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    ####################on ferme les trous dans les GR pour que watershed marche bien###################################
    for cnt in contours:
        #cv2.drawContours(opened, [cnt], 0,255,-1, cv2.LINE_8)
        cv2.drawContours(thresh, [cnt], 0, 255, -1)
    # plt.imshow(thresh,cmap='gray')
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    for label in np.unique(labels):
        #si label ==0 on regarde le fond à ignorer
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
    
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        print(c)
        x, y, w, h = cv2.boundingRect(c)
        gr = gray[y:y + h, x:x + w]
        plt.imshow(gr)
        plt.title("gr")
        plt.show()
        cv2.rectangle(gray, (x, y), (x + w, y + h), (200, 0, 100), 2)
    
    plt.imshow(gray)
    plt.title("Detection")
    plt.show()
    
detect(io.imread('images_thomas/nouveau_champ_IMG_2350.jpg'))