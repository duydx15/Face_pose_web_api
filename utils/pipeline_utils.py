import cv2 
import numpy as np 
import os 
import time 


FACEMESH_lips_1 = [212, 432]

def check_landmark(retina_landmark, facemesh_landmark, coordinate_bbox, H, W):
    '''
    params: retina_landmark: numpy array (2*2)
    params: facemesh_landmark: numpy array (2*2)
    Rule: retina landmark like as groundtruth
    If facemesh landmark 2 keypoint in lips far away from retina landmark return False
    else return True
    
    '''
    topleft_bbox, _ = coordinate_bbox
    l_facemesh_keypoint = []
    for i in range(len(FACEMESH_lips_1)):
        idx = FACEMESH_lips_1[i]
        x = facemesh_landmark.landmark[idx].x
        y = facemesh_landmark.landmark[idx].y

        realx = int(x * W)
        realy = int(y * H)
        l_facemesh_keypoint.append((realx, realy))
    l_lip_retina_keypoint = []
    for keypoint in retina_landmark[3:]:
        x = keypoint[0] - topleft_bbox[0]
        y = keypoint[1] - topleft_bbox[1]
        l_lip_retina_keypoint.append((x,y))
    
    
    THRESHOLD_center_distance = 0.1 * H
    THRESHOLD_lengthlip_ratio = 1
    y_center_retina = int((l_lip_retina_keypoint[0][1] + l_lip_retina_keypoint[1][1])/2)
    
    y_center_facemesh = int((l_facemesh_keypoint[0][1] + l_facemesh_keypoint[1][1])/2)
    length_lip_retina = abs(l_lip_retina_keypoint[1][0] - l_lip_retina_keypoint[0][0]) + 0.0001
    length_lip_facemesh = abs(l_facemesh_keypoint[1][0] - l_facemesh_keypoint[0][0])
    if abs(y_center_facemesh - y_center_retina) > THRESHOLD_center_distance:
        return False
    if length_lip_facemesh/length_lip_retina <= THRESHOLD_lengthlip_ratio:
        return False
    return True
    