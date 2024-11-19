import torch
import os
import re
import time
import torch
import logging
from filelock import FileLock
from typing import List, Optional
from dataclasses import dataclass
from transformers import BertTokenizer
from torch.utils.data.dataset import Dataset
from ratsnlp.nlpbook.ner import NERTrainArguments

from ratsnlp import nlpbook
from ratsnlp.nlpbook.ner import NERCorpus, NERDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import SequentialSampler
from transformers import BertConfig, BertForTokenClassification
from ratsnlp.nlpbook.ner import NERTask


args = NERTrainArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_corpus_root_dir = "chatbot/chatbotServer/chatbot_conv/ner/",
    downstream_corpus_name="data",
    downstream_model_dir="chatbot/chatbotServer/chatbot_conv/ner/model/",
    batch_size=32 if torch.cuda.is_available() else 4,
    learning_rate=1e-5,
    max_seq_length=20,
    epochs=5,
    seed=7,
)


nlpbook.set_seed(args)
nlpbook.set_logger(args)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)



corpus = NERCorpus(args)

train_dataset = NERDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="train",
)


train_dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=RandomSampler(train_dataset, replacement=False),
    collate_fn=nlpbook.data_collator,
    drop_last=False,
)

val_dataset = NERDataset(
    args=args,
    corpus=corpus,
    tokenizer=tokenizer,
    mode="val",
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    sampler=SequentialSampler(val_dataset),
    collate_fn=nlpbook.data_collator,
    drop_last=False,

)

pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=corpus.num_labels,
)
model = BertForTokenClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
)

###task 정의
task = NERTask(model, args)

###trainer 정의
trainer = nlpbook.get_trainer(args)

#학습 시작
trainer.fit(
    task,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
