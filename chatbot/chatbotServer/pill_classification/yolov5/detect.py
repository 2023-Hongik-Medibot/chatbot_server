
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
import os
import sys
from pathlib import Path
import torch
import numpy as np
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative #### detect.py 바로 위의 경로. yolov5 폴더 위치가 됨.

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (Profile,check_img_size,check_requirements, non_max_suppression, print_args, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode



def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im


@smart_inference_mode()
def run(
        weights,  # model path or triton URL
        image,
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):  
    h,w,_ = image.shape
    if h==4032 and w == 3042:
        pass
    else:
        new_image = np.zeros((4032,3042,3),dtype='uint8')
        if h>4032 and w>3042:
            ratio = max(h/4032,w/3042)
            image = cv2.resize(image,(int(w/ratio),int(h/ratio)))
        if h>4032 and w<=3042:
            ratio = max(h/4032,w/3042)
            image = cv2.resize(image,(int(w/ratio),int(h/ratio)))
        if h<=4032 and w>3042:
            ratio = max(h/4032,w/3042)
            image = cv2.resize(image,(int(w/ratio),int(h/ratio)))
        if h<4032 and w<3042:
            ratio = min(4032/h,3042/w)
            image = cv2.resize(image,(int(w*ratio),int(h*ratio)))
        h,w,_ = image.shape
        hd = int((4032-h)/2)
        wd = int((3042-w)/2)

        new_image[hd:hd+h,wd:wd+w] = image
        image = new_image
        
        


    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride = model.stride
    pt = model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = ['pill','타원형','원형']


    im0s = image
    im = letterbox(im0s, imgsz, stride = stride, auto= pt) # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    ###im은 설정한 크기에 맞게 resize(padding)된 이미지(array)
    ###im0s은 원래 크기의 이미지(array)

    bs = 1  # batch_size
   
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    dt =(Profile(), Profile(), Profile())
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        # Inference
    with dt[1]:
        pred = model(im, augment=augment, visualize=visualize)
        # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions
    for det in pred:  # per image
        im0 = im0s.copy()
        if len(det):
                # Rescale boxes from img_size to im0 size
                ########################변형 전 사이즈      박스    변형 후 사이즈
                ########################변형시킨 이미지로부터 반환된 바운딩 박스를 변환 전 사이즈에 맞는 바운딩 박스 크기로 되돌려줌. 
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
            conf_list = []
            for *xyxy, conf, cls in reversed(det):
              conf_list.append(conf)
                
            one_cls = 0
            max_conf = -1
            for i in range(len(conf_list)):
              if conf_list[i]>max_conf:
                one_cls = i
                max_conf = conf_list[i]
            i = 0
            for *xyxy, conf, cls in reversed(det):
                if i == one_cls:
                    cl= int(cls)  # integer class
                      #return (names[c], xyxy)
                      ###xyxy는 바운딩 박스의 좌표일 것으로 추축됨!
                    x1,y1,x2,y2 =int(xyxy[0]) ,int(xyxy[1]) ,int(xyxy[2]) ,int(xyxy[3] )
                    image = np.array(im0[y1:y2, x1:x2])
                    h,w,_ = image.shape
                    c = max(h,w)
                    ratio = 448/c
                    h = int(h*ratio)
                    w = int(w*ratio)
                    image = cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)
                    wm = int((448-w)//2)
                    hm = int((448-h)//2)
                    back = np.zeros((448,448,3),np.uint8)
                    back[hm:hm+h,wm:wm+w] = image
                    output= pp2(back)
                    return (names[cl],output)
                i+=1
        else:
            return (False,False)

def pp2(image):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  _,_,gray = cv2.split(image)
  sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  gray = cv2.filter2D(gray, -1, sharpening_mask1) #### 선명하게 
  gray = cv2.add(gray,-40) ###밝기 낮추기
  clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)) ###경계 뚜렷하게 만들기
  gray = clahe.apply(gray)
  gray = cv2.GaussianBlur(gray, (0, 0), 1)
  gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
  return gray
