import torch.nn as nn
import torch
from pytorch_pretrained_bert import BertModel
from transformers import RobertaModel
class EnsembleClassifer(nn.Module):
    def __init__(self,h1,h2,drop_out_rate):
        super(EnsembleClassifer,self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(drop_out_rate);
        self.linear_1 = nn.Linear(1536,120)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(120,2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,tokens1,masks1,tokens2,masks2):
        _, pooled_output = self.bert(tokens1, attention_mask=masks1, output_all_encoded_layers=False)
        output = self.roberta(tokens2,masks2)[1]
        next_output = torch.concat([pooled_output,output],dim = 1)
        next_output = self.dropout(next_output)
        next_output = self.linear_1(next_output)
        next_output = self.relu(next_output)
        next_output = self.linear_2(next_output)
        proba = self.sigmoid(next_output)
        return proba