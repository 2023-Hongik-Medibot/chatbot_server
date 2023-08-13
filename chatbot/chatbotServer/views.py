from django.shortcuts import render
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from chatbotServer.chatbot_conv.intent_classification.model.model_test import predict_sentiment
from chatbotServer.chatbot_conv.ner.model.model_test import medi,medi_store, hospital
# Create your views here.

from django.http import HttpResponse

def index(request):
    return HttpResponse("Hello world")

@method_decorator(csrf_exempt,name="dispatch") #csrf 무시 decorator
def intent_and_ne(request):
    data=json.loads(request.body.decode('utf-8'))
    ask = data["ask"]
    intent = str(predict_sentiment(ask))
    if intent == "0" or intent == "3" or intent =="4" or intent=="5":
        ne = medi(ask)
    if intent =="1":
        ne = medi_store(ask)
    if intent=="2":
        ne = hospital(ask)
    
    return JsonResponse({"intent": intent, "entity":ne})
    