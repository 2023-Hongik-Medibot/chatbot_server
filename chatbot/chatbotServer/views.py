from django.shortcuts import render
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from chatbotServer.chatbot_conv.intent_classification.model.model_test import predict_sentiment
from chatbotServer.chatbot_conv.ner.model.model_test import medi,medi_store, hospital
from chatbotServer.pill_classification.pill_classify import classify
import boto3
import cv2
import os
# Create your views here.

from django.http import HttpResponse
from django.conf import settings


file1 = "img1.jpg"
file2 = "img2.jpg"
bucket_name= settings.BUCKET_NAME
region = settings.REGION
access_key=settings.ACCESS_KEY
secret_access_key=settings.SECRET_ACCESS_KEY


client = boto3.client('s3',
                      aws_access_key_id=access_key,
                      aws_secret_access_key=secret_access_key,
                      region_name=region)


def index(request):
    return HttpResponse("Hello world")

@method_decorator(csrf_exempt,name="dispatch") #csrf 무시 decorator
def intent_and_ner(request):
    data=json.loads(request.body.decode('utf-8'))
    ask = data["ask"]
    if '"' in ask:
        ask = ask.replace('"','',2)
    intent = str(predict_sentiment(ask))
    if intent == "0" or intent == "3" or intent =="4" or intent=="5":
        ne = medi(ask)
    if intent =="1":
        ne = medi_store(ask)
    if intent=="2":
        ne = hospital(ask)

    
    return JsonResponse({"intent": intent, "entity":ne})
    
@method_decorator(csrf_exempt,name="dispatch") #csrf 무시 decorator
def pill_classification(request):
    data = json.loads(request.body.decode('utf-8'))
    image1 = data["image"][0]["key"]
    client.download_file(Bucket = bucket_name,Key = image1, Filename =file1)
    image1 = cv2.imread(file1)
    os.remove(file1)
    image2 = data["image"][1]["key"]
    client.download_file(Bucket = bucket_name,Key = image2, Filename =file2)
    image2 = cv2.imread(file2)
    os.remove(file2)
    result = classify(image1,image2)
    if result == None:
        result = []
    return JsonResponse({"result" : result})
