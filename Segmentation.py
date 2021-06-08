# -*- coding: utf-8 -*-

#####################################################################################################
# Module :          Segmentation of blood red cells            
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
from copy import copy
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

import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)                        
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        cv2.rectangle(im0,(x1, y1),(x2, y2),(0,255,0),1)
                        res = copy(det)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


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
    # labels = skimage.segmentation.watershed(-D, markers, mask=thresh)
    labels = watershed(-D, markers, mask=thresh)
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
print_img(splits[3], "split")
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

