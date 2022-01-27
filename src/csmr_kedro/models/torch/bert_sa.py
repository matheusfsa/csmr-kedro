import math
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
from transformers import AutoConfig

from csmr_kedro.extras.datasets import COMPANIES

RANDOM_STATE = 42

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, drop=0.1, layer_norm=True, n_classes=3):
        super().__init__()
        self.use_layer_norm = layer_norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop)
        self.classifier = nn.Linear(hidden_size, n_classes)
        
    def forward(self, features, **kwargs):
        if self.use_layer_norm:
          features = self.layer_norm(features)
        x = self.dropout(features)
        logits = self.classifier(x)
        return logits

class BertSA(nn.Module):
    def __init__(self, 
                 drop=0.2, 
                 layer_norm=False, 
                 seq_length=128, 
                 hidden_size=64,
                 predict_company=False,
                 company_alpha=0.4,
                 use_company=False,
                 company_dim=32):
        super(BertSA, self).__init__()
        model_name = 'neuralmind/bert-base-portuguese-cased'
        #pretrained
        #model_name = "/content/drive/MyDrive/companies_sm_rep/modelling/pt_tweets_bert/checkpoint-400"
        self.config = AutoConfig.from_pretrained(model_name)  

        self.bert = AutoModel.from_pretrained(model_name)
        
        self.hidden_size = self.config.hidden_size
        self.use_company = use_company
        if self.use_company:
          self.company_embedding = nn.Embedding(len(COMPANIES), company_dim)
          self.classifier = ClassificationHead(self.hidden_size + company_dim, drop=drop, layer_norm=layer_norm, n_classes=3)
        else:
          self.classifier = ClassificationHead(self.hidden_size, drop=drop, layer_norm=layer_norm, n_classes=3)
        self.predict_company = predict_company
        
        if self.predict_company:
          self.company_predictor = ClassificationHead(self.hidden_size, drop=drop, layer_norm=layer_norm, n_classes=len(COMPANIES))
          self.company_criterion = nn.CrossEntropyLoss()
          self.company_alpha = company_alpha
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.sa_criterion = nn.CrossEntropyLoss()
        
        
    
    def embedding(self, input_ids, companies_ids):

        bert_output = self.bert(input_ids=input_ids)
        x = bert_output[1]
        return x 

    def forward(self, input_ids, labels, company_id=None, attention_mask=None):
        bert_output = self.bert(input_ids=input_ids)
        x = bert_output[1]
        if self.use_company:
          cx = self.company_embedding(company_id)
          x = torch.cat((x, cx), dim=1)
        logits = self.classifier(x)
        
        output = {}
        output["logits"] = logits
          
        if labels is not None:
          output["loss"] = self.sa_criterion(logits, labels)
          if self.predict_company:
            comp_logits = self.company_predictor(x)
            output["loss"] += self.company_alpha*self.company_criterion(comp_logits, company_id)
            
        return output