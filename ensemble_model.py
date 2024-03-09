from utils.data_loading import load_data
from preprocesser.clean_text_new import preprocess,text_preprocessing_pipeline,dataframefn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils.loaders import BertData,RoBERTaData,DistilBertData
from utils.train_ensemble_function import train_ensemble
from utils.valid_ensemble_function import validate
from models.ENSEMBLE import EnsembleClassifer
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_snippets import *
from torchsummary import summary

batch_size = 8 
word_max_len = 64
h1 = 768
h2 = 128
drop_out_rate = 0.2
epochs = 20
learning_rate = 3e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = load_data()
data["text"] = data["text"].apply(lambda x : preprocess(x))
data = data[data['text'].apply(lambda x: len(x) > 0)]
'''gpt_data["text"] = gpt_data["text"].apply(lambda x : preprocess(x))
gpt_data = gpt_data[gpt_data['text'].apply(lambda x: len(x) > 0)]'''
data["text"] = data["text"].apply(lambda x : text_preprocessing_pipeline(x))
data = data[data['text'].apply(lambda x: len(x) > 0)]
'''gpt_data["text"] = gpt_data["text"].apply(lambda x : text_preprocessing_pipeline(x))
gpt_data = gpt_data[gpt_data['text'].apply(lambda x: len(x) > 0)]'''
X_train,X_valid,y_train,y_valid = dataframefn(data);
class_list = [0,1]
class_num = len(class_list)
train_dataset = BertData(X_train, y_train, word_max_len)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,
                      collate_fn=train_dataset.collate_fn)
validate_dataset = BertData(X_valid, y_valid, word_max_len)
validate_sampler = SequentialSampler(validate_dataset)
validate_dataloader = DataLoader(validate_dataset, sampler=validate_sampler, batch_size=batch_size,
                    collate_fn=validate_dataset.collate_fn)
train_dataset_roberta = RoBERTaData(X_train, y_train, word_max_len)
train_sampler_roberta = RandomSampler(train_dataset_roberta)
train_dataloader_roberta = DataLoader(train_dataset_roberta, sampler=train_sampler_roberta, batch_size=batch_size,
                      collate_fn=train_dataset_roberta.collate_fn)
validate_dataset_roberta = RoBERTaData(X_valid, y_valid, word_max_len)
validate_sampler_roberta = SequentialSampler(validate_dataset_roberta)
validate_dataloader_roberta = DataLoader(validate_dataset_roberta, sampler=validate_sampler_roberta, batch_size=batch_size,
                    collate_fn=validate_dataset_roberta.collate_fn)
e_c = EnsembleClassifer(1,2,0.1)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(e_c.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
for epoch in range(epochs):
    n_batch = len(train_dataloader)
    for i,(data_1,data_2) in enumerate(zip(train_dataloader,train_dataloader_roberta)):
        train_loss, train_acc = train_ensemble(data_1,data_2, e_c, 
                                      optimizer, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print("pos=",pos, "train_loss=",train_loss, "train_acc=",train_acc, end='\r')
        
    total_predict = []
    total_label = []

    n_batch = len(validate_dataloader)
    for i,(data_1,data_2) in enumerate(zip(validate_dataloader,validate_dataloader_roberta)):
        val_loss, val_acc = validate(data_1,data_2, e_c, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print("pos=",pos, "val_loss=",val_loss, "val_acc=",val_acc, end='\r')
    
    scheduler.step()