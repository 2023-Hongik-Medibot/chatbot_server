import sys
sys.path.append('.')
import torch
from torchtext import data
import pandas as pd
from chatbot.chatbotServer.chatbot_conv.config.globalParam import MAX_LEN,EMBEDDING_DIM
import torch.nn as nn
import torch.optim as optim
from chatbot.chatbotServer.chatbot_conv.intent_classification.ChatbotNet1 import ChatbotNet1
import numpy as np
from konlpy.tag import Okt,Twitter


#####사전 읽어오기.
def read_vocab(path):
    vocab = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            index, token = line.split('\t')
            token,_ = token.split('\n')
            vocab[token] = int(index)
    return vocab

def stopwordList():

    stop_words=[]
    with open('chatbot/chatbotServer/chatbot_conv/intent_classification/stopword.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word= line.strip('\n')
            stop_words.append(word)
    return stop_words

def tokenizer(text):
    stop_words = stopwordList()
    okt = Okt()
    text = okt.morphs(text,stem=True)
    result = [] 
    for t in text: 
        if t not in stop_words: 
            result.append(t)
    return result

def predict_sentiment(sentence):
        vocab = read_vocab('chatbot/chatbotServer/chatbot_conv/intent_classification/dict_vocab.tsv')
        torch.no_grad()
        tokenized = [token for token in tokenizer(sentence)]
        if len(tokenized) < MAX_LEN:
            tokenized += ['<pad>'] * (MAX_LEN - len(tokenized))
        indexed = []
        print(tokenized)
        for token in tokenized:
            if token not in vocab.keys(): #### 토큰이 단어 사전에 없으면 단어 사전에 추가해줌.
                indexed.append(0)
            else:
                indexed.append(vocab[token])
        print(indexed)
        tokens = []
        for i in range(MAX_LEN):  
            if(indexed[i]==0):
                if(i!=0 and indexed[i-1]==0):
                    pass
                else :
                    tokens.append(indexed[i])
            else :
                tokens.append(indexed[i])
        padding = MAX_LEN - len(tokens)
        for i in range(padding):
            tokens.append(1)

        print(tokens)
        tensor = torch.LongTensor(tokens)
        tensor = tensor.unsqueeze(0)
        model = ChatbotNet1(len(vocab),EMBEDDING_DIM,MAX_LEN)
        model.load_state_dict(torch.load("chatbot/chatbotServer/chatbot_conv/intent_classification/train_state/intent_state.pt"))
        model.eval()
        out = model(tensor)
        print(out)
        out = torch.argmax(out,dim=0)
        out = out.item()
        intent = out
        return intent

intent= ["주의사항 묻기","약국 정보 묻기", "병원 정보 묻기","약 부작용 묻기","약 효과 묻기","약 복용법 묻기"]
sen = "영등포역 근처 신경 정신과 알려주세요"
print(intent[predict_sentiment(sen)])