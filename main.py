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
from datetime import datetime
from segmentation import *
from zip_mod import zip_folder, histo_results, list_files
import shutil

# #####################################################################################################
# # EXECUTION                                    
# #####################################################################################################
empty_folder(['results/splits', 'results/segmented', 'results/boxes'])
now = datetime.now()
now_str = now.strftime("%d/%m/%Y %H:%M:%S")
print("\n------------------IMAGE SELECTION & SPLIT------------------")
#Selection of image
print("SELECTION OF IMAGE :")
imgss = select_imgs()
for imgs in imgss :
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
    print("\n------------------SEGMENTATION------------------")
    #-----Start Timer
    start_time = time.time()
    #-----Segmentation of red blood cells
    print("SEGMENTATION OF RED BLOOD CELLS :")
    start_time_step = time.time()
    splits_1 = copy.deepcopy(splits)
    boxes_contour = get_boxes_contour(splits_1, splits, window_size, margin, img_name)
    boxes_contour = list(nms(np.array(boxes_contour), 0.46))
    splits_2 = copy.deepcopy(splits)
    splits_path = save_splits(splits, 'results/splits/', img_name)
    boxes_model = get_boxes_model(splits_path, splits_2, window_size, margin, img_name)
    boxes_model = list(nms(np.array(boxes_model), 0.46))
    boxes = boxes_contour + boxes_model
    # boxes = boxes_contour
    print("     ", "Number of boxes before NMS :", len(boxes))
    end_time_step = time.time()
    print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))
    #-----Elimination of duplicate segmentations
    print("\nELIMINATION OF DUPLICATE SEGMENTATIONS :")
    start_time_step = time.time()
    boxes = list(nms(np.array(boxes), 0.46))
    boxes = clean_boxes(boxes)
    print("     ", "Number of boxes after NMS :", len(boxes))
    histo_results('results/results.xlsx', img_name, now_str, len(boxes_contour), len(boxes_model), len(boxes))
    end_time_step = time.time()
    print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))
    #-----Save boxes
    # print("\nSAVE BOXES :")
    # start_time_step = time.time()
    # save_boxes(copy.deepcopy(img), boxes, 'combined', img_name)
    # end_time_step = time.time()
    # print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))
    #-----Draw boxes
    print("\nDRAW BOXES :")
    start_time_step = time.time()
    merged = draw_boxes(copy.deepcopy(img), boxes_contour)
    cv2.imwrite('results/segmented/' + img_name + '_contour.jpg', merged)
    print_img(merged, "Segmented by Contour Detection")
    
    merged = draw_boxes(copy.deepcopy(img), boxes_model)
    cv2.imwrite('results/segmented/' + img_name + '_model.jpg', merged)
    print_img(merged, "Segmented by Model")
    merged = draw_boxes(copy.deepcopy(img), boxes)
    
    cv2.imwrite('results/segmented/' + img_name + '_combined.jpg', merged)
    print_img(merged, "Segmented by Contour Detection + Model")
    end_time_step = time.time()
    print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))

    #-----End Timer
    end_time = time.time()
    print("\nTEMPS EXECUTION CONTOUR DETECTION : %s secondes" % (end_time - start_time))
# print("\nEVALUATE BOXES :")
# start_time_step = time.time()
# imgs_path = list_files('results/boxes/')
# evaluate_model(imgs_path)
# # shutil.move('results/boxes.zip', 'results/boxes/boxes.zip')
# end_time_step = time.time()
# print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))
    
# print("\nZIP BOXES :")
# start_time_step = time.time()
# zip_folder('results/boxes.zip', 'results/boxes/', True)
# # shutil.move('results/boxes.zip', 'results/boxes/boxes.zip')
# end_time_step = time.time()
# print("     ", "TEMPS EXECUTION : %s secondes" % (end_time_step - start_time_step))

    