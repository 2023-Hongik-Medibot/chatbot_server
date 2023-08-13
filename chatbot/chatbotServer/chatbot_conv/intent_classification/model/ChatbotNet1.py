import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import torch


class ChatbotNet1(nn.Module):
  
    
    def calculate_features(self):
        out_conv1 = (self.seq_length  - 1 * self.kernel[0]) + 1
        out_conv1 = math.floor(out_conv1)
        out_pool1 = (out_conv1 - 1 * (self.kernel[0])) + 1
        out_pool1 = math.floor(out_pool1)

        out_conv2 = (out_pool1  - 1 * self.kernel[1]) + 1
        out_conv2 = math.floor(out_conv2)
        out_pool2 = (out_conv2 - 1 * self.kernel[1]) + 1
        out_pool2 = math.floor(out_pool2)
      
        out = (out_pool2) * self.output_size #torch.cat이후 최종 size
        return out



    def __init__(self, vocab_size, embedding_size, seq_length):
        super(ChatbotNet1,self).__init__()
        self.seq_length = seq_length #sequence_length(이전 TEXT Field에서 정의한 fix_length값) 
        self.embedding_size = embedding_size
        self.kernel = [2,3,4]#### conv filter로  총 3개의 필터, 2,3,4 사이즈의 필터를 사용.
        self.output_size = 32
        self.padding = seq_length
       

# ### embedding
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_size)
        self.fc = nn.Linear(in_features=19, out_features=6,bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)
#         ### 입력 정수(사전)에 대해서 밀집 벡터로 변환하는 계층
#         ##num_embeddings = 임베딩 벡터를 사용할 전체 범주 개수, embedding_dim = 임베딩 백터의 차원 , padding_idx = 여기까지 패딩!

# # ##### convoulutin layer
#         self.conv1 = nn.Conv1d(in_channels = self.embedding_size,out_channels =self.output_size,kernel_size = self.kernel[0],stride=1, padding=self.padding) #seq_length, out_seq,kernel_size
#         self.conv2 = nn.Conv1d(in_channels = self.embedding_size,out_channels =self.output_size,kernel_size = self.kernel[1],stride=1,padding=self.padding)
# #         self.conv3 = nn.Conv1d(in_channels = self.embedding_size,out_channels =self.output_size,kernel_size = self.kernel[2],stride=1,padding=self.padding)
# #         #### 100,128,2
# # #####  pooling layer
#         self.pool1 = nn.MaxPool1d(self.kernel[0],stride = 1)
#         self.pool2 = nn.MaxPool1d(self.kernel[1],stride = 1)
# #         self.pool3 = nn.MaxPool1d(self.kernel[2],stride = 1)

# # ###### fooly connected layer
#         self.linear1 = nn.Linear(self.calculate_features(),256)
#         self.linear2 = nn.Linear(256,128)
#         self.linear3 = nn.Linear(128,6)


        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels= self.embedding_size,out_channels= 1,kernel_size= self.kernel[0]), ###20-1 = 19
            nn.ReLU(),
            nn.MaxPool1d(self.kernel[0]), ###19/2 = 9
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels= self.embedding_size,out_channels= 1,kernel_size= self.kernel[1]), ###20 - 2 = 18 
            nn.ReLU(),
            nn.MaxPool1d(self.kernel[1]) ### 18 / 3 = 6
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv1d(in_channels= self.embedding_size,out_channels= 1,kernel_size= self.kernel[2]), ###20 - 3 = 17 
            nn.ReLU(),
            nn.MaxPool1d(self.kernel[2]) ### 17 / 4 = 4
        )




    def forward(self,seq):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        x = self.embedding(seq) ###x->(batch size,embedding_dim, seq_length)
       
        x = x.transpose(1,2)
  
        # x1 = self.conv1(x)  ### -> (batch size, 1 , seq_length - filter_size+1)
        # x1 = F.relu(x1)
        # x1 = self.pool1(x1)

        # x2 = self.conv2(x1)  ### -> (batch size, 1 , seq_length - filter_size+1)
        # x2 = F.relu(x2)
        # x2 = self.pool2(x2)
    
        out1 = self.conv_layer1(x)
        out2 = self.conv_layer2(x)
        out3 = self.conv_layer3(x)

        # # x2 = self.conv2(x1)
        # # x2 = F.relu(x2)
        # # x2 = self.pool2(x2)
       
        # # x3 = self.conv3(x2)
        # # x3 = F.relu(x3)
        # # x3 = self.pool3(x3)
  
        x_concat = torch.cat((out1,out2,out3),dim=2) #2번째 차원 기준으로 묶음.(32,30,17) + (32,30,13) => (32,30,30)
        # x = torch.flatten(x,1) #batch를 제외한 나머지를 묶어버린다 -> FC layer를 사용하기 위함. (32,30,30) -> (32,900)
        # out = self.linear1(x)
        # out = self.linear2(out)
        # out = self.linear3(out)
        out = self.fc(x_concat)
        out = out.squeeze()
        return out





    # self.conv_layers = nn.Sequential(
    #   nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim),

    #   nn.Conv2d(1, 32, (3,embedding_dim)),
    #   nn.ReLU(),
    #   nn.MaxPool1d(kernel_size=2,stride=2)
    # )
    # result_size = int((max_len-filter_sizes+1)/2)
    # self.Linear = nn.Linear(result_size, output_dim)
   

#   def forward(self, x):

#     x = self.conv_layers(x)
#     x = nn.Flatten(x)
#     logits = self.Linear(x)
#     return logits

    

