import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch

from craft.model import CRAFT
from craft.util import parse_region_map
from yolov5.detect import run


device = 'cuda' if torch.cuda.is_available() else "cpu"


def draw_box(img, boxes):
    img = cv2.resize(img, (224,224))
    for box in boxes:
        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255))
    cv2.imshow("f",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

####글자 순서대로 정렬
def sorting(boxes):
    for i in range(len(boxes)):
        min = boxes[i]
        min_num = i
        for j in range(i+1, len(boxes)):
            l1,r1,u1,d1 = boxes[j][1],boxes[j][3],boxes[j][0],boxes[j][2]
            l2,r2,u2,d2 = min[1],min[3],min[0],min[2]
            x1,y1 = int(l1 + (r1-l1)//2), int(u1 + (d1-u1)//2)
            x2,y2 = int(l2 + (r2-l2)//2), int(u2 + (d2-u2)//2)
            d = (d1-u1 + d2-u2)//4
    
            if x1<x2 and abs(y1-y2) < d:
                min = boxes[j]
                min_num = j
            elif x1<x2 and y1 +d <y2:
                min = boxes[j]
                min_num = j
            elif x1> x2 and y1 + d <y2:
                min = boxes[j]
                min_num = j
            else:
                pass
        boxes[i],boxes[min_num] = boxes[min_num],boxes[i]
    return boxes



def text_detection(image,state):
    img = torch.Tensor(image)
    img = img.permute(2,0,1).unsqueeze(0).to(device) 
    craft = CRAFT().to(device)
    craft.load_state_dict(torch.load(state,map_location=torch.device('cpu')))
    craft.eval()
    out = craft(img).squeeze(0)
    out = np.array(out.to('cpu').detach().numpy())
    # cv2.imshow("f",out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    boxes = parse_region_map(out,150)
    # draw_box(image,boxes)
    boxes = sorting(boxes)
    image = cv2.resize(image,(224,224))
    output_boxes = []
    for box in boxes:
        l,r,u,d = box[1],box[3],box[0],box[2]
        if l<0 :
            l=0
        if l>223:
            l = 223
        if r<0 :
            r=0
        if r>223:
            r = 223
        if u<0 :
            u=0
        if u>223:
            u = 223
        if d<0 :
            d=0
        if d>223:
            d = 223

        w,h = r-l,d-u
        wd,hd = int((224-w)/2), int((224-h)/2)
        detected_box = image[u:d,l:r]
        output = np.zeros((224,224,3),dtype=np.uint8)
        output[hd:hd+h, wd:wd+w] = detected_box
        
        output_boxes.append(output)
    return output_boxes


            