# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:13:40 2021

@author: thomas.collaudin
"""

#####################################################################################################
# Module :          Combined methods (Contour detection + YOLO model)           
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
#----------------------------------------------------------------------------------------------------
# GENERAL
#----------------------------------------------------------------------------------------------------
# import Segmentation as seg
from Segmentation import *

# #####################################################################################################
# # EXECUTION                                    
# #####################################################################################################
# # empty_folder(['2_splits', '3_segmented', '4_boxes'])

print("\n------------------IMAGE SELECTION & SPLIT------------------")
#Selection of image
print("SELECTION OF IMAGE :")
imgs = select_img()
img_file, img, img_w, window_size, margin = imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]
img_name = img_file[img_file.rfind('/') + 1 : -4]
print_img(img, "Source")
print("     ", img_file)
#Split of image
print("\nSPLIT OF IMAGE :")
start_time_step = time.time()
splits = split_image(img, window_size, margin)
print_split(splits, "Splitted")
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))

#Segmentation by contour detection
print("\n------------------SEGMENTATION BY CONTOUR DETECTION------------------")
#-----Start Timer
start_time = time.time()
#-----Segmentation of red blood cells
print("SEGMENTATION OF RED BLOOD CELLS :")
start_time_step = time.time()
splits_1 = copy.deepcopy(splits)
boxes_contour = get_boxes_contour(splits_1, splits, window_size, margin, img_name)

#-----Saving splits
# print("\nSAVING SPLITS :")
start_time_step = time.time()
splits_2 = copy.deepcopy(splits)
splits_path = save_splits(splits, '2_splits/', img_name)
# print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))
boxes_model = get_boxes_model(splits_path, splits_2, window_size, margin, img_name)
boxes = boxes_contour + boxes_model

print("     ", "Number of boxes before NMS :", len(boxes))
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))
#-----Elimination of duplicate segmentations
print("\nELIMINATION OF DUPLICATE SEGMENTATIONS :")
start_time_step = time.time()
boxes = list(nms(np.array(boxes), 0.46))
print("     ", "Number of boxes after NMS :", len(boxes))
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))
#-----Save boxes
print("\nSAVE BOXES :")
start_time_step = time.time()
save_boxes(copy.deepcopy(img), boxes, 'combined', img_name)
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))
#-----Draw boxes
print("\nDRAW BOXES :")
start_time_step = time.time()
merged = draw_boxes(copy.deepcopy(img), boxes)
cv2.imwrite('3_segmented/' + img_name + '_combined.png', merged)
print_img(merged, "Segmented by Contour Detection + Model")
end_time_step = time.time()
print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))
#-----End Timer
end_time = time.time()
print("\nTEMPS EXECUTION CONTOUR DETECTION : %s secondes" % (end_time - start_time))