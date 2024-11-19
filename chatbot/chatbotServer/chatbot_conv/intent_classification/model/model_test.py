import sys
sys.path.append('.')
import torch
from torchtext import data
import pandas as pd
from chatbotServer.chatbot_conv.config.globalParam import MAX_LEN,EMBEDDING_DIM
import torch.nn as nn
import torch.optim as optim
from chatbotServer.chatbot_conv.intent_classification.model.ChatbotNet1 import ChatbotNet1
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
    with open('chatbotServer/chatbot_conv/intent_classification/stopword.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word= line.strip('\n')
            stop_words.append(word)
    return stop_words

def tokenizer(text):
    stop_words = stopwordList()
    okt = Okt()
    text = okt.morphs(text,stem = True) ##stem = True
    result = [] 
    for t in text: 
        if t not in stop_words: 
            result.append(t)
    return result

def predict_sentiment(sentence):
        vocab = read_vocab('chatbotServer/chatbot_conv/intent_classification/dict_vocab2.tsv')
        torch.no_grad()
        tokenized = [token for token in tokenizer(sentence)]
        print(tokenized)
        if len(tokenized) < MAX_LEN:
            tokenized += ['<pad>'] * (MAX_LEN - len(tokenized))
        indexed = []
        for token in tokenized:
            if token not in vocab.keys(): #### 토큰이 단어 사전에 없으면 단어 사전에 추가해줌.
                indexed.append(0)
            else:
                indexed.append(vocab[token])
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
        tensor = torch.LongTensor(tokens)
        tensor = tensor.unsqueeze(0)
        model = ChatbotNet1(len(vocab),EMBEDDING_DIM,MAX_LEN)
        model.load_state_dict(torch.load("chatbotServer/chatbot_conv/intent_classification/train_state/intent_state_stem_true.pt"))
        model.eval()
        out = model(tensor)
        out = torch.argmax(out,dim=0)
        out = out.item()
        intent = out
        intents = ["주의사항","약국","병원","부작용","효능","방법"]
        print(intents[int(intent)])
        return intent

