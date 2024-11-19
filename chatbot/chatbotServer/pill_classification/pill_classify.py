
from chatbotServer.pill_classification.yolov5.detect import run
from chatbotServer.pill_classification.craft.infer import text_detection
from chatbotServer.pill_classification.deep_text_recognition_benchmark.infer import infer
import cv2
import numpy as np
import pymysql
import openpyxl
import pandas as pd
import numpy as np
from django.conf import settings

###글자 교정
def replace_text(input_text):
    texts = []
    texts.append(input_text)
    for i in range(len(input_text)):
        new_texts = []
        if input_text[i] == 'o':
            for text in texts:
                text = text[:i] + '0' + text[i+1:]
                new_texts.append(text)
            # texts.extend(new_texts)
        elif input_text[i] == '0':
            for text in texts:
                text = text[:i] + 'o' + text[i+1:]
                new_texts.append(text)
            #texts.extend(new_texts)
        elif input_text[i] == '1':
            for text in texts:
                text = text[:i] + 'i' + text[i+1:]
                new_texts.append(text)
        elif input_text[i] == 'i':
            for text in texts:
                text = text[:i] + '1' + text[i+1:]
                new_texts.append(text)
        elif input_text[i] == 'e':
            for text in texts:
                text = text[:i] + 'f' + text[i+1:]
                new_texts.append(text)
        elif input_text[i] == 'f' :
            for text in texts:
                text = text[:i] + 'e' + text[i+1:]
                new_texts.append(text)
        texts.extend(new_texts)
    return texts


def find_pill(shapes,text1,text2):
    pills = []
    text11 = replace_text(text1)
    text22 = replace_text(text2)
    user = settings.DB_USER
    pw = settings.DB_PASSWORD
    dbname = settings.DB_NAME
    host = settings.DB_HOST
    db = pymysql.connect(host=host, port=3306, user=user, passwd=pw,db =dbname, charset='utf8')
    cursor = db.cursor()
    for text1 in text11:
        for text2 in text22 :
            for shape in shapes:
                if text1 == '' and text2 !='' :
                    sql = "SELECT * from pill WHERE shape='%s' AND (text_front ='%s' OR text_back ='%s')" % (shape,text2,text2)
                elif text2 == '' and text1 !='':
                    sql = "SELECT * from pill WHERE shape='%s' AND (text_front ='%s' OR text_back ='%s')" % (shape,text1,text1)
                elif text1 == '' and text2 == '':
                    db.close()
                    return None
                else :
                    sql = "SELECT * from pill WHERE shape='%s' AND ((text_front ='%s' AND text_back ='%s') OR (text_front ='%s' AND text_back ='%s'))"% (shape,text1,text2,text2,text1)
                cursor.execute(sql)
                results = cursor.fetchall() 
                if len(results):
                    for result in results:
                        pills.append(result[2])                    
    if len(pills):
        db.close()
        if len(pills) > 5:
            return pills[:4]
        else : 
            return pills
    
    for text1 in text11 :
        for shape in shapes:
            sql = "SELECT * from pill WHERE shape='%s' AND (text_front ='%s' OR text_back ='%s')" % (shape,text1,text1)
            cursor.execute(sql)
            results = cursor.fetchall() 
            if len(results):
                for result in results:
                    pills.append(result[2])
   
    for text2 in text22 :
        for shape in shapes:
            sql = "SELECT * from pill WHERE shape='%s' AND (text_front ='%s' OR text_back ='%s')" % (shape,text2,text2)
            cursor.execute(sql)
            results = cursor.fetchall() 
            if len(results):
                for result in results:
                    pills.append(result[2])

    db.close()
    if len(pills):
        if len(pills) > 5:
            return pills[:4]
        else : 
            return pills
    return None
    

def pp2(image):

  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  mean_brightness = np.mean(gray)/255
  brightness = int((1-mean_brightness) * -30)
  gray = cv2.add(gray,brightness)
  clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 10))
  gray = clahe.apply(gray)
  return gray


def recognition(image):
    ##image는 경로 또는 바이너리

    output = run(weights = "chatbotServer/pill_classification/yolov5/weights/best.pt", image = image)
    shape = output[0]
    image = output[1]
    if shape != False:
  
        state = "chatbotServer/pill_classification/craft/epoch_150_new.pt"
        boxes = text_detection(image,state)
        character = ""
        for box in boxes:
            img = cv2.cvtColor(pp2(box),cv2.COLOR_GRAY2RGB) ###두번째 recognition 학습했을 때!
            result = infer(img)
            character = character + result
        if character ==None:
            character = ''
        return (shape,character)
    else : 
        return False
    
def classify(image1, image2):

    recognition1 =  recognition(image1)
    recognition2 = recognition(image2)
    if recognition1 == False:
        shape1 =''
        text1 = ''
    else : 
        shape1 = recognition1[0]
        text1 = recognition1[1]

    if recognition2 == False:
        shape2 = ''
        text2 = ''
    else:
        shape2 = recognition2[0]
        text2 = recognition2[1]

    if shape1 != shape2:
        shape = ['타원형','원형','장방형']
    else:
        if shape1 == '타원형':
            shape = ['타원형','장방형']
        else : 
            shape = ['원형']
    print(shape,text1, text2)
    pills = find_pill(shape,text1, text2)
    return pills


