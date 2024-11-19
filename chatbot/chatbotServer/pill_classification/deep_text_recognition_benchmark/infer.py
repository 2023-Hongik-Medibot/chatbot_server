
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from yolov5.detect import run
from craft.infer import text_detection
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from deep_text_recognition_benchmark.myutils import  AttnLabelConverter
from deep_text_recognition_benchmark.model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Opt():
    def __init__(self):
        super(Opt,self).__init__()
        self.Transformation = 'TPS'
        self.FeatureExtraction ='ResNet'
        self.SequenceModeling = 'BiLSTM'
        self.Prediction = 'Attn'
        self.imgH =224
        self.imgW = 224
        self.input_channel = 3
        self.num_class =43
        self.num_fiducial = 20
        self.output_channel = 512
        self.hidden_size = 256
        self.batch_max_length = 25
        self.rgb = True
        self.character = "0123456789분할선마크abcdefghijklmnopqrstuvwxyz/-"
        self.saved_model = "chatbotServer/pill_classification/deep_text_recognition_benchmark/saved_models/best_accuracy2.pth"
        self.PAD = True
        self.batch_size = 1
        self.workers = 0




def infer(image):
    opt = Opt()
    """ model configuration """
    
    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    ##### 학습용 이미지로 전환하기 array to tensor
    image = Image.fromarray(image).convert('RGB')
    interpolation = Image.BICUBIC
    toTensor = transforms.ToTensor()
    image = image.resize((224,224), interpolation)
    image = toTensor(image)
    image = image.sub_(0.5).div_(0.5).unsqueeze(0)

    image_tensors =image

    model.eval()
    with torch.no_grad():
        
        batch_size = image_tensors.size(0)
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)


        preds = model(image, text_for_pred, is_train=False)
            # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        pred = preds_str[0]
        pred_max_prob = preds_max_prob[0]

                
        pred_EOS = pred.find('[s]')
        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
        pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
        return (pred)
 
     
    