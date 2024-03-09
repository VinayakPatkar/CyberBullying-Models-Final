import torch.nn as nn
from pytorch_pretrained_bert import BertModel
class Bert_Aggression_Identification_Model(nn.Module):
    def __init__(self, h1, h2, class_num, drop_out_rate):
        super(Bert_Aggression_Identification_Model, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.dropout = nn.Dropout(drop_out_rate)
        self.linear1 = nn.Linear(h1, h2)
        self.linear2 = nn.Linear(h2, class_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, tokens, masks):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        d = self.dropout(pooled_output)
        x = self.relu(self.linear1(d))
        proba = self.softmax(self.linear2(x))
        
        return proba