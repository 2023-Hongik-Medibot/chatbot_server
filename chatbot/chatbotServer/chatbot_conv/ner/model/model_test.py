from ratsnlp.nlpbook.ner import NERDeployArguments
from transformers import BertForTokenClassification
import torch
from transformers import BertTokenizer
from transformers import BertConfig


args = NERDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="chatbotServer/chatbot_conv/ner/model/",
    downstream_model_checkpoint_fpath ="chatbotServer/chatbot_conv/ner/model/epoch=4-val_loss=0.01.ckpt" ,
    max_seq_length=20,
)
####사용하고자 하는 체크포인트 state값이 있으면 ,downstream_model_checkpoint_fpath 인자로 전달
#### downstream_model_dir에 그냥 체크포인트 경로만 넣어주면 NERDeployArguments가 알아서 쳌포인트 경로중에서 val_loss가 가장 작은 쳌포인트를 골라서 downstream_model_checkpoint_fpath인자에 경로를 전달함.


tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)


fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)


pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)
model = BertForTokenClassification(pretrained_model_config)


#코드8 체크포인트 읽기
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})


##코드9 EVAL MODE
model.eval()

labels = [label.strip() for label in open(args.downstream_model_labelmap_fpath, "r").readlines()]
id_to_label = {}
for idx, label in enumerate(labels):
  if "MED" in label:
    label = "MED"
  elif "LOC" in label:
    label = "LOC"
  elif "HOS" in label:
    label = "HOS"
  else:
    label = label
  id_to_label[idx] = label

#####문장의 각 토큰에 대한 예측 개체명을 반환
def inference_fn(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        probs = outputs.logits[0].softmax(dim=1)
        top_probs, preds = torch.topk(probs, dim=1, k=1)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_tags = [id_to_label[pred.item()] for pred in preds]
        result = []
        for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
            if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
                token_result = {
                    "token" :token,
                    "tag":predicted_tag
                    #"top_prob": str(round(top_prob[0].item(), 4)),
                }
                result.append(token_result)
    return {
        "sentence": sentence,
        "result": result,
    }




# 1. 약국 검색 -> 지역명
# 2. 병원 검색 -> 지역명, 병원
# 3. 약 검색 -> 약 이름

def hospital(text) :
    loc = []
    hos = []
    result = inference_fn(text)
    results = result['result']
    tokens = result['sentence'].split()
    tag_list_per_word = []
    tag_list = []
    for result in results:
        if '##' not in result['token']:
            tag_list_per_word.append(tag_list)
            tag_list = []
            tag_list.append(result['tag'])
        else :
            tag_list.append(result['tag'])
    tag_list_per_word.append(tag_list)
    tag_list_per_word.remove([])

    for num,tag in enumerate(tag_list_per_word):
        if 'MED' in tag or 'LOC' in tag : 
           loc.append(tokens[num])
        if 'HOS' in tag : 
           hos.append(tokens[num])
    for num,l in enumerate(loc) : 
        if "랑" in l: 
           temp,_ = l.split('랑')
           if _ == '':
              loc[num] = temp
        elif "이랑" in l:
           temp,_ = l.split('이랑')
           if _ == '':
              loc[num] = temp
        elif "와" in l :
           temp,_ = l.split('와')
           if _ == '':
              loc[num] = temp
        elif "과" in l :
           temp,_ = l.split('과')
           if _ == '':
              loc[num] = temp
        elif "에서" in l:
           temp,_ = l.split('에서')
           if _ == '':
              loc[num] = temp
        elif "에" in l:
           temp,_ = l.split('에')
           if _ == '':
              loc[num] = temp
    if len(hos) == 0:
       hos.append("병원")
    return {"장소" : loc, "병원" : hos}
       

def medi_store(text) :
    loc = []
    result = inference_fn(text)
    results = result['result']
    tokens = result['sentence'].split()
    tag_list_per_word = []
    tag_list = []
    for result in results:
        if '##' not in result['token']:
            tag_list_per_word.append(tag_list)
            tag_list = []
            tag_list.append(result['tag'])
        else :
            tag_list.append(result['tag'])
    tag_list_per_word.append(tag_list)
    tag_list_per_word.remove([])

    for num,tag in enumerate(tag_list_per_word):
        if 'MED' in tag or 'LOC' in tag : 
           loc.append(tokens[num])
    for num,l in enumerate(loc) : 
        if "랑" in l: 
           temp,_ = l.split('랑')
           if _ == '':
              loc[num] = temp
        elif "이랑" in l:
           temp,_ = l.split('이랑')
           if _ == '':
              loc[num] = temp
        elif "와" in l :
           temp,_ = l.split('와')
           if _ == '':
              loc[num] = temp
        elif "과" in l :
           temp,_ = l.split('과')
           if _ == '':
              loc[num] = temp
        elif "에서" in l:
           temp,_ = l.split('에서')
           if _ == '':
              loc[num] = temp
        elif "에" in l:
           temp,_ = l.split('에')
           if _ == '':
              loc[num] = temp
    
    return loc
       

       
def medi(text) :
    medi = []
    result = inference_fn(text)
    results = result['result']
    print(results)
    tokens = result['sentence'].split()
    tag_list_per_word = []
    tag_list = []
    for result in results:
        if '##' not in result['token']:
            tag_list_per_word.append(tag_list)
            tag_list = []
            tag_list.append(result['tag'])
        else :
            tag_list.append(result['tag'])
    tag_list_per_word.append(tag_list)
    tag_list_per_word.remove([])

    for num,tag in enumerate(tag_list_per_word):
        if 'MED' in tag or 'LOC' in tag : 
           medi.append(tokens[num])
    for num,l in enumerate(medi) : 
        if "의" in l: 
           temp,_ = l.split('의')
           if _ == '':
              medi[num] = temp
        elif "를" in l:
           temp,_ = l.split('를')
           if _ == '':
              medi[num] = temp
        elif "을" in l :
           temp,_ = l.split('을')
           if _ == '':
              medi[num] = temp
        elif "은" in l :
           temp,_ = l.split('은')
           if _ == '':
              medi[num] = temp
        elif "는" in l:
           temp,_ = l.split('는')
           if _ == '':
              medi[num] = temp
        
    
    return medi


