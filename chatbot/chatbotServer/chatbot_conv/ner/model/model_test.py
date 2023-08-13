from ratsnlp.nlpbook.ner import NERDeployArguments
from transformers import BertForTokenClassification
import torch
from transformers import BertTokenizer
from transformers import BertConfig


args = NERDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="chatbotServer/intent_model/ner/model/",
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

def medi_store(text):
    result = inference_fn(text)["result"]
    loc = {}
    num = 0
    for i in result:
        if i["tag"] =="LOC":
            if "##" not in i["token"]: ### 처음에 걸림
                loc[num]=i["token"]
            else:
                loc[num] = i["token"]
                for j in range(10):
                    index_forward = num - j-1
                    if "##" in result[index_forward]["token"]:
                        loc[index_forward] = result[index_forward]["token"]
                    else:
                        loc[index_forward] = result[index_forward]["token"]
                        break
                for j in range(10):
                    index_backward = num - j+1
                    if "##" in result[index_forward]["token"]:
                        loc[index_forward] = result[index_backward]["token"]
                    else:
                        break  
        num=num+1
    loc = list(loc.values())
    num =0
    loc_list=["","",""]
    for tok in loc:
        if "##" not in tok:
            loc_list[num] = tok
            num = num+1
        else:
            loc_list[num-1]+=tok.replace("##",'')

    loc_list = [i for i in loc_list if i!='']
    return loc_list


def hospital(text):
    result = inference_fn(text)["result"]
    loc = {}
    hos={}
    num = 0
    for i in result:
        if i["tag"] =="LOC":
            if "##" not in i["token"]: ### 처음에 걸림
                loc[num]=i["token"]
            else:
                loc[num] = i["token"]
                for j in range(10):
                    index_forward = num - j-1
                    if "##" in result[index_forward]["token"]:
                        loc[index_forward] = result[index_forward]["token"]
                    else:
                        loc[index_forward] = result[index_forward]["token"]
                        break
                for j in range(10):
                    index_backward = num - j+1
                    if "##" in result[index_forward]["token"]:
                        loc[index_backward] = result[index_backward]["token"]
                    else:
                        break  
        if i["tag"] =="HOS":
            if "##" not in i["token"]: ### 처음에 걸림
                hos[num]=i["token"]
            else:
                hos[num] = i["token"]
                for j in range(10):
                    index_forward = num - j-1
                    if "##" in result[index_forward]["token"]:
                        hos[index_forward] = result[index_forward]["token"]
                    else:
                        hos[index_forward] = result[index_forward]["token"]
                        break
                for j in range(10):
                    index_backward = num - j+1
                    if "##" in result[index_forward]["token"]:
                        hos[index_backward] = result[index_backward]["token"]
                    else:
                        break 
        num=num+1
    loc = list(loc.values())
    hos = list(hos.values())
    num =0
    loc_list=['','','']
    for tok in loc:
        if "##" not in tok:
            loc_list[num] = tok
            num = num+1
        else:
            loc_list[num-1]+=tok.replace("##",'')
    hos_list=['','','']
    for tok in hos:
        if "##" not in tok:
            hos_list[num] = tok
            num = num+1
        else:
            hos_list[num-1]+=tok.replace("##",'')
    loc_list = [i for i in loc_list if i!='']
    hos_list = [i for i in hos_list if i!='']
    return {"장소":loc_list,"병원":hos_list}

def medi(text):
    result = inference_fn(text)["result"]
    med = {}
    num = 0
    print(result)
    for i in result:
        if i["tag"] =="MED":
            if "##" not in i["token"]: ### 처음에 걸림
                med[num]=i["token"]
            else:
                med[num] = i["token"]
                for j in range(10):
                    index_forward = num - j-1
                    if "##" in result[index_forward]["token"]:
                        med[index_forward] = result[index_forward]["token"]
                    else:
                        med[index_forward] = result[index_forward]["token"]
                        break
                for j in range(10):
                    index_backward = num - j+1
                    if "##" in result[index_forward]["token"]:
                        med[index_forward] = result[index_backward]["token"]
                    else:
                        break  
        num=num+1
    med = list(med.values())
    num =0
    med_list=['','','']
    for tok in med:
        if "##" not in tok:
            med_list[num] = tok
            num = num+1
        else:
            med_list[num-1]+=tok.replace("##",'')

    med_list = [i for i in med_list if i!='']
    return med_list

# text = "바세린 부작용을 알려줘"
# print(inference_fn(text)["result"])

# result = inference_fn(text)["result"]
# medi(result)