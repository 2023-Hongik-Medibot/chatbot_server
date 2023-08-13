import random
# origin = open('chatbot/chatbotServer/intent_model/ner/data/chatbot_dataset.txt', 'r')
# label = open('chatbot/chatbotServer/intent_model/ner/data/chatbot_dataset_label.txt', 'r')
# result = open('chatbot/chatbotServer/intent_model/ner/data/chatbot_dataset_concat.txt', 'w')

# original  =  origin.readlines()
# modi = label.readlines()
# num = len(original)

# for i in range(num):
    
#     result.write(original[i].replace('\n', '')+'\u241E'+modi[i])

# origin.close()
# label.close()
# result.close()

###### test data, train data 나누기.

concat= open('chatbot/chatbotServer/intent_classification/ner/data/chatbot_dataset_concat.txt', 'r')
train = open('chatbot/chatbotServer/intent_model/ner/data/train.txt', 'w')
test = open('chatbot/chatbotServer/intent_model/ner/data/val.txt', 'w')


result = concat.readlines()

print(type(result))

random.shuffle(result)

print(type(result))

train_set = result[300:]
val_set = result[:300]

for i in train_set:
    train.write(i)

for i in val_set:
    test.write(i)
    
concat.close()
train.close()
test.close()

train = open('chatbot/chatbotServer/intent_model/ner/data/train.txt', 'r')

train_data = train.readlines()
hos_num = 0
loc_num = 0
med_num=0

for i in train_data:
    if "HOS" in i:
        hos_num+=1
    if "MED" in i:
        med_num+=1
    if "LOC" in i:
        loc_num+=1
print(hos_num,loc_num, med_num)

train.close()