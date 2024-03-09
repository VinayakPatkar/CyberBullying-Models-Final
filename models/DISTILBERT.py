from transformers import DistilBertModel
import torch.nn as nn
class DistilBert_Aggression_Identification_Model(nn.Module):
    def __init__(self,h1,h2,class_num,drop_out_rate):
        super(DistilBert_Aggression_Identification_Model,self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(drop_out_rate)
        self.linear1 = nn.Linear(h1, h2)
        self.linear2 = nn.Linear(h2, class_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    def forward(self, tokens, masks):
        pooled_output = self.distilbert(tokens, attention_mask=masks)[0][:,0,:]
        d = self.dropout(pooled_output)
        x = self.relu(self.linear1(d))
        proba = self.softmax(self.linear2(x))
        return proba