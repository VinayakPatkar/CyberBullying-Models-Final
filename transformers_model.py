from utils.basic_model_pipeline import pipeline
from utils.data_loading import load_data
from preprocesser.clean_text_new import preprocess,text_preprocessing_pipeline,dataframefn
from utils.train_function import train
from utils.valid_function import validate
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils.loaders import BertData,RoBERTaData,DistilBertData
from models.BERT import Bert_Aggression_Identification_Model
from models.ROBERTA import RoBERTa_Aggression_Identification_Model
from models.DISTILBERT import DistilBert_Aggression_Identification_Model
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
model = Bert_Aggression_Identification_Model(h1, h2, class_num, drop_out_rate).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
for epoch in range(epochs):
    n_batch = len(train_dataloader)
    for i, data in enumerate(train_dataloader):
        train_loss, train_acc = train(data, model, 
                                      optimizer, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print("pos=",pos, "train_loss=",train_loss, "train_acc=",train_acc, end='\r')
        
    total_predict = []
    total_label = []

    n_batch = len(validate_dataloader)
    for i, data in enumerate(validate_dataloader):
        val_loss, val_acc = validate(data, model, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print("pos=",pos, "val_loss=",val_loss, "val_acc=",val_acc, end='\r')
    
    scheduler.step()
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f"trained_models/BERT_{learning_rate}_{batch_size}.pth")


train_dataset_roberta = RoBERTaData(X_train, y_train, word_max_len)
train_sampler_roberta = RandomSampler(train_dataset_roberta)
train_dataloader_roberta = DataLoader(train_dataset_roberta, sampler=train_sampler_roberta, batch_size=batch_size,
                      collate_fn=train_dataset_roberta.collate_fn)
validate_dataset_roberta = RoBERTaData(X_valid, y_valid, word_max_len)
validate_sampler_roberta = SequentialSampler(validate_dataset_roberta)
validate_dataloader_roberta = DataLoader(validate_dataset_roberta, sampler=validate_sampler_roberta, batch_size=batch_size,
                    collate_fn=validate_dataset_roberta.collate_fn)
model = RoBERTa_Aggression_Identification_Model(h1, h2, class_num, drop_out_rate).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(epochs):
    n_batch = len(train_dataloader_roberta)
    for i, data in enumerate(train_dataloader_roberta):
        train_loss, train_acc = train(data, model, 
                                      optimizer, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print("pos=",pos, "train_loss=",train_loss, "train_acc=",train_acc, end='\r')
        
    total_predict = []
    total_label = []

    n_batch = len(validate_dataloader_roberta)
    for i, data in enumerate(validate_dataloader_roberta):
        val_loss, val_acc = validate(data, model, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print("pos=",pos, "val_loss=",val_loss, "val_acc=",val_acc, end='\r')
    
    scheduler.step()
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f"trained_models/ROBERTA_{learning_rate}_{batch_size}.pth")


train_dataset_distil = DistilBertData(X_train, y_train, word_max_len)
train_sampler_distil = RandomSampler(train_dataset_distil)
train_dataloader_distil = DataLoader(train_dataset_distil, sampler=train_sampler_distil, batch_size=batch_size,
                      collate_fn=train_dataset_distil.collate_fn)
validate_dataset_distil = DistilBertData(X_valid, y_valid, word_max_len)
validate_sampler_distil = SequentialSampler(validate_dataset_distil)
validate_dataloader_distil = DataLoader(validate_dataset_distil, sampler=validate_sampler_distil, batch_size=batch_size,
                    collate_fn=validate_dataset_distil.collate_fn)
model = DistilBert_Aggression_Identification_Model(h1, h2, class_num, drop_out_rate).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(epochs):
    n_batch = len(train_dataloader_distil)
    for i, data in enumerate(train_dataloader_distil):
        train_loss, train_acc = train(data, model, 
                                      optimizer, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print("pos=",pos, "train_loss=",train_loss, "train_acc=",train_acc, end='\r')
        
    total_predict = []
    total_label = []

    n_batch = len(validate_dataloader_distil)
    for i, data in enumerate(validate_dataloader_distil):
        val_loss, val_acc = validate(data, model, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        print("pos=",pos, "val_loss=",val_loss, "val_acc=",val_acc, end='\r')
    
    scheduler.step()
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, f"trained_models/DISTILBERT_{learning_rate}_{batch_size}.pth")
