import sys
sys.path.append('.')
import torch
from torchtext import data
from torchtext.data import TabularDataset
from torchtext.data import Iterator
import pandas as pd
from chatbot.chatbotServer.chatbot_conv.config.globalParam import MAX_LEN, EMBEDDING_DIM 
import torch.nn as nn
import torch.optim as optim
from ChatbotNet1 import ChatbotNet1
import numpy as np
from konlpy.tag import Okt , Komoran,Hannanum


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device} device")
    

df = pd.read_csv('chatbot/chatbotServer/chatbot_conv/intent_classification/raw_data/chatbot_dataset2.csv')
df = df.sample(frac=1)
train_df = df[400:]
test_df  = df[:400]
train_df.to_csv("chatbot/chatbotServer/chatbot_conv/intent_classification/train_data/train_data_chatbot2.csv", index=False)
test_df.to_csv("chatbot/chatbotServer/chatbot_conv/intent_classification/train_data/test_data_chatbot2.csv", index=False)

stop_words=[]
with open('chatbot/chatbotServer/chatbot_conv/intent_classification/stopword.txt', 'r', encoding='utf-8') as f:
    for line in f:
        word= line.strip('\n')
        stop_words.append(word)

def tokenizer(text):
    okt = Okt()
    tokens = okt.morphs(text,stem =True)
    token=[]
    for tok in tokens:
        if(tok not in stop_words):
            token.append(tok)
    # print(token)
    # print(text)
    
    return token


        ## 필드란 텐서로 표현 될 수 있는 텍스트 데이터 타입을 처리한다. 필드를 통해 앞으로 어떤 전처리를 할지 정의할 수 있다.
# tokenizer = Okt()
TEXT = data.Field(tokenize = tokenizer,
                batch_first = True,
                fix_length = MAX_LEN)
LABEL = data.Field(
                sequential=False,
                use_vocab=False,
                batch_first=False,
                is_target=True)

        # sequential : 순차 데이터 여부. False이면 토큰화가 적용되지 않음. (default: True)
        # use_vocab : Vocab 개체 사용 여부. False인 경우 이 필드의 데이터는 이미 숫자여야 함. (default: True)
        # tokenize : 사용될 토큰화 함수 (default: string.split)
        # lower : 영어 데이터 소문자화 (default: False)
        # batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부 (default: False)
        # is_target : target variable 여부 (default: False)
        # fix_length : 최대 허용 길이. 이 길이에 맞춰 패딩(Padding) 작업 진행


train_data, test_data = TabularDataset.splits(
        path='chatbot/chatbotServer/chatbot_conv/intent_classification/train_data/', train='train_data_chatbot2.csv', test='test_data_chatbot2.csv', format='csv',
        fields=[('text', TEXT),('label', LABEL)], skip_header = True)

print("fdfd")
TEXT.build_vocab(train_data, min_freq=1, max_size=10000) ### data에 대한 사전 생성. + embedding 과정.
print("ㅇㅁㅅㅁ")

def save_vocab(vocab, path):
    with open(path, 'w+', encoding='utf-8') as f:     
        for token, index in vocab.stoi.items():
            f.write(f'{index}\t{token}\n')

save_vocab(TEXT.vocab,'chatbot/chatbotServer/chatbot_conv/intent_classification/dict_vocab2.tsv')



## iterator 생성.
batch_size = 32
train_loader = Iterator(dataset=train_data, batch_size = batch_size)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)


####모델 생성
model = ChatbotNet1(len(TEXT.vocab),EMBEDDING_DIM,MAX_LEN)

######optimzier 정보

optimizer = optim.Adam(model.parameters(),lr=1e-3) 
criterion = nn.CrossEntropyLoss()


#### train

for epoch in range(50):
            ##### train
    model.train() ### 너 이제 학습 시킬거라고 말해주는거다..
    current_loss = 0.0
            ### batch_num은 현재까지지 얼마만큼의의 data가가 불려졌는지
            #### x와와 target은은 x와와 label이다
    for batch_num,(x,target) in enumerate(train_loader):
        optimizer.zero_grad() #### 한번 학습이 완료될 때 마다 gradient를를 0으로 초기화해줌
        x,target = x.to(device), target.to(device)
        out = model(x) #### model에 train_data를 넣었을 때 output 반환
        target = target.long()
        loss = criterion(out,target) #### 결과값과 실제제 값 사이 loss를 계산
            
            ###########Back propagation 단계 시작!##############
            
        loss.backward() #### 모든 가중치에 대해서 loss값 미분
        optimizer.step() #### 가중치 수정(신기하게도 parameter 없이도 동작함...뭐를 저장하고 있어서 가능한거라,,,모름)
        current_loss += loss    
            ####### 5개의 배치를 훈련했을 때 총 loss를 출력함.
        if (batch_num+1)%5 == 0 or (batch_num+1)%5 == len(train_loader):
                print('epoch:%d,batch_num:%d,current_loss:%.3f' %(epoch,batch_num+1,current_loss/100))
                current_loss = 0.0
        #####test 
    with torch.no_grad(): #### to exclude test data from training -> autoGrad를 끔
        model.eval() ### model한테 이제부터 evlautate할거라고 알려
        total_samples = 0.0
        correct_samples = 0.0
        for (x,target) in test_loader:
            x,target = x.to(device),target.to(device)
            out = model(x) 
            pred = torch.argmax(out,1) #### model을 거쳐서 1*10의 output이 나옴. 가장 큰 값의 index를 반환
            correct_samples+= (pred==target).sum()
        accuracy = 100*float(correct_samples)/float(len(test_data))
    print('accuracy:%.3f' %(accuracy))
    if epoch%5==0:
        torch.save(model.state_dict(), 'chatbot/chatbotServer/chatbot_conv/intent_classification/train_state/intent_state_'+str(epoch)+"_"+str(accuracy)+'.pt')
        print("save")