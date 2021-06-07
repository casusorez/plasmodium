# -*- coding: utf-8 -*-

#####################################################################################################
# Module :          Segmentation of a blood sample image by contour detection of red cells            
#----------------------------------------------------------------------------------------------------
# Author :          Thomas Collaudin
#----------------------------------------------------------------------------------------------------
# Date:             1st June 2021                      
#----------------------------------------------------------------------------------------------------
# Project :         Detection and identification of plasmodiums on microscopic images     
#----------------------------------------------------------------------------------------------------
# PhD Student :     Aniss Acherar                     
#----------------------------------------------------------------------------------------------------
# PhD Supervisor :  Renaud Piarroux                     
#####################################################################################################

#####################################################################################################
# ADDITIONAL MODULES                            
#####################################################################################################
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import filedialog

from scipy import ndimage
from skimage.morphology import watershed
import skimage
from skimage.feature import peak_local_max
from skimage import color
from skimage import io

from sklearn.cluster import KMeans
import random
from statistics import mean

import time
from numba import jit, int32
from pprint import pprint

import warnings
warnings.filterwarnings("ignore")

#####################################################################################################
# FUNCTIONS                                    
#####################################################################################################

#----------------------------------------------------------------------------------------------------
# Function :        split_image
# Role :            Split image in sub images of window_size * window_size dimensions
#----------------------------------------------------------------------------------------------------
def split_image(img, window_size, margin):
    sh = list(img.shape)
    sh[0], sh[1] = sh[0] + margin * 2, sh[1] + margin * 2
    img_ = np.zeros(shape=sh)
    img_[margin:-margin, margin:-margin] = img
    stride = window_size
    step = window_size + 2 * margin
    nrows, ncols = img.shape[0] // window_size, img.shape[1] // window_size
    splitted = []
    # Split
    for i in range(nrows):
        for j in range(ncols):
            h_start = j * stride
            v_start = i * stride
            cropped = img_[v_start:v_start + step, h_start:h_start + step]
            splitted.append(cropped.astype('uint8'))
    return splitted

#----------------------------------------------------------------------------------------------------
# Function :        merge_image
# Role :            Merge splitted images in one image
#----------------------------------------------------------------------------------------------------
def merge_image(splitted, margin):
    sh = list(splitted[0].shape)
    sh[0], sh[1] = sh[0] - margin * 2, sh[1] - margin * 2
    nrows, ncols = int(np.sqrt(len(splitted))), int(np.sqrt(len(splitted)))
    merged = np.zeros(shape=[sh[0] * nrows, sh[1] * ncols, 3])
    # Merge
    for i in range(nrows):
        for j in range(ncols):
            merged[i * sh[0] : (i + 1) * sh[0], j * sh[1] : (j + 1) * sh[1]] = splitted[j + i * ncols][margin : margin + sh[0], margin : margin + sh[1]]
    return merged.astype('uint8')

#----------------------------------------------------------------------------------------------------
# Function :        detect
# Role :            Contours detection using watershed algorithm
#----------------------------------------------------------------------------------------------------
def detect(img) :
    gray = np.uint8(color.rgb2gray(img) * 255)
    #gray c'est ton image en niveaux de gris
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 3) #3 ou 2 max 
    thresh = cv2.dilate(erosion,kernel,iterations = 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    #on ferme les trous dans les GR pour que watershed marche bien
    for cnt in contours:
        #cv2.drawContours(opened, [cnt], 0,255,-1, cv2.LINE_8)
        cv2.drawContours(thresh, [cnt], 0, 255, -1)
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=10, labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = skimage.segmentation.watershed(-D, markers, mask=thresh)
    rects = list()
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
        x, y, w, h = cv2.boundingRect(c)
        # gr = gray[y:y + h, x:x + w]
        cv2.rectangle(gray, (x, y), (x + w, y + h), (200, 0, 100), 2)
        rects.append([x, y, x + w, y + h])
    return rects

#----------------------------------------------------------------------------------------------------
# Function :        kmeans
# Role :            Classification of pixels in 3 clusters using kmeans algorithm
#----------------------------------------------------------------------------------------------------
def kmeans(img):
    x,y,c = img.shape
    kmeans = KMeans(n_clusters=3, n_init=100).fit(img.reshape(x*y, c))

    for l in range(len(kmeans.labels_)) :
        for c in range(3) :
            if kmeans.labels_[l] == 2:
                img[int(l / x)][int(l % x)][c] = 0
            else :
                img[int(l / x)][int(l % x)][c] =  255
    return img

#----------------------------------------------------------------------------------------------------
# Function :        segmentation
# Role :            White staining of red blood cells
#----------------------------------------------------------------------------------------------------
@jit(nopython=True)
def segmentation(img, moyennes, n) :
    for x in range(len(img)) :   
        for y in range(len(img[0])) :  
            moyenne = int((img[x][y][0] + img[x][y][1] + img[x][y][2]) / 3)
            ecart = 1000
            for m in range(len(moyennes)) :
                if abs(moyennes[m] - moyenne) < ecart :
                    label = m
                    ecart = abs(moyennes[m] - moyenne)
            if n == 3 :
                if label == 1 :
                    img[x][y] = [255, 255, 255] 
                else :
                    img[x][y] = [0, 0, 0]
            elif n == 2 :
                if label == 0 : 
                    img[x][y] = [255, 255, 255] 
                else :
                    img[x][y] = [0, 0, 0]
    return img

#----------------------------------------------------------------------------------------------------
# Function :        nms
# Role :            Elimination of duplicate segmentations
#----------------------------------------------------------------------------------------------------
def nms(boxes, overlapThresh):
# if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
	# initialize the list of picked indexes	
    pick = []
	# grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]    
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
    while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                        np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
    return boxes[pick].astype("int")

#----------------------------------------------------------------------------------------------------
# Function :        select_img
# Role :            Selection of an image
#----------------------------------------------------------------------------------------------------
def select_img() -> filedialog.Open:
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    img_file = askopenfilename(parent=root)
    root.destroy
    
    img = cv2.imread(img_file)
    img_w = img.shape[0]
    window_size = int(img_w / 4)
    margin = 40
    return [img_file, img, img_w, window_size, margin]

#----------------------------------------------------------------------------------------------------
# Function :        print_img
# Role :            Print of an image
#----------------------------------------------------------------------------------------------------
def print_img(img, title) :
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.grid(False)
    plt.show()
    
#----------------------------------------------------------------------------------------------------
# Function :        print_split
# Role :            Print of splitted images
#----------------------------------------------------------------------------------------------------
def print_split(splits, title) : 
    fig=plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.axis('off')
    for i in range(1, len(splits) + 1):
        fig.add_subplot(4, 4, i)
        plt.imshow(splits[i - 1])
        plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.grid(False)
    plt.show()

#----------------------------------------------------------------------------------------------------
# Function :        get_boxes
# Role :            Elimination of duplicate segmentations
#----------------------------------------------------------------------------------------------------
def get_boxes(splits) :
    boxes = list()
    for s in range(len(splits)) :
        split = splits[s]
        if s not in [5, 6, 9, 10] :
            nb_clusters = 3
        else : 
            nb_clusters = 2
            
        pts = list()
        for p in range(10000) : 
            x = random.randrange(0, len(split))
            y = random.randrange(0, len(split[0]))
            pts.append(split[x][y])
        km = KMeans(n_clusters=nb_clusters, n_init=100).fit(pts)
        
        rgbs = [[] for _ in range(nb_clusters)]
        for l in range(len(km.labels_)) :    
            rgbs[km.labels_[l]].append(pts[l])
            
        moyennes = [[0, 0, 0] for _ in range(nb_clusters)]
        for l in range(len(moyennes)) :
            for c in range(3) :
                moyennes[l][c] = mean([rgbs[l][k][c] for k in range(len(rgbs[l]))])
        
        for m in range(len(moyennes)) :
            moyennes[m] = mean(moyennes[m])
        moyennes.sort()
        
        split = segmentation(split, moyennes, nb_clusters)
        row, col = int(s / 4), int(s % 4)    
        for box in detect(split) :
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            w, h = x2 - x1, y2 - y1
            x1 += col * window_size - margin
            y1 += row * window_size - margin
            boxes.append([int(x1), int(y1), int(x1) + w, int(y1) + h])  
    return boxes

#----------------------------------------------------------------------------------------------------
# Function :        draw_boxes
# Role :            Draw blue boxes on an image
#----------------------------------------------------------------------------------------------------
def draw_boxes(img, boxes) :
    for box in boxes :
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        if x2 - x1 > 29 and x2 - x1 < 100 and y2 - y1 > 29 and y2 - y1 < 100 :
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img
    
#####################################################################################################
# EXECUTION                                    
#####################################################################################################

#Selection of image
print("SELECTION OF IMAGE :")
imgs = select_img()
img_file, img, img_w, window_size, margin = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]
print_img(img, "Source")
print("     ", img_file)

#Start Timer
start_time = time.time()

#Split of image
print("\nSPLIT OF IMAGE :")
start_time_step = time.time()
splits = split_image(img, window_size, margin)
print_split(splits, "Splitted")
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))

#Segmentation of red blood cells
print("\nSEGMENTATION OF RED BLOOD CELLS :")
start_time_step = time.time()
boxes = get_boxes(splits)
print("     ", "Number of boxes :", len(boxes))
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))

#Elimination of duplicate segmentations
print("\nELIMINATION OF DUPLICATE SEGMENTATIONS :")
start_time_step = time.time()
boxes = list(nms(np.array(boxes), 0.46))
print("     ", "Number of boxes after NMS :", len(boxes))
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))

#Merge splitted images
print("\nMERGE SPLITTED IMAGES :")
start_time_step = time.time()
merged = merge_image(splits, margin)
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))

#Draw boxes
print("\nDRAW BOXES :")
start_time_step = time.time()
merged = draw_boxes(img, boxes)
print_img(img, "Segmented and Merged")
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))

#End Timed
end_time = time.time()

print("\nTEMPS EXECUTION GLOBAL : %s secondes" % (end_time - start_time))

