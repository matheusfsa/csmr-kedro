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

class CompanyAttention(nn.Module):
    def __init__(self, n_companies, dim, hidden_dim, dropout,num_attention_heads=2, seq_length=128):
      super(CompanyAttention, self).__init__()
      self.company_embedding = nn.Embedding(n_companies, dim)
      self.seq_length = seq_length
      self.num_attention_heads = num_attention_heads
      self.attention_head_size = int(dim / num_attention_heads)
      self.all_head_size = self.num_attention_heads * self.attention_head_size

      self.query = nn.Linear(dim, self.all_head_size) # the current focus of attention when being compared to all of the other query preceding inputs
      self.key = nn.Linear(hidden_dim, self.all_head_size) # a preceding input being compared to the current focus of attention
      self.value = nn.Linear(hidden_dim, self.all_head_size) # used to compute the output for the current focus of attention
      self.scaler = 1/np.sqrt(self.attention_head_size)
      self.dropout = nn.Dropout(dropout)
      self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
      new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
      x = x.view(*new_x_shape)
      if len(x.size()) < 4:
        x = x.unsqueeze(-1)
        return x.permute(0, 1, 3, 2)
      return x.permute(0, 2, 1, 3)

    def transpose_query(self, x):
      new_x_shape = (x.size()[0], self.seq_length, int(x.size()[-1]/self.seq_length))
      x = x.view(*new_x_shape)
      return x

    def forward(self, hidden_states, company, return_att=False):
      company = self.company_embedding(company)

      query_layer =self.query(company)
      query_layer = self.transpose_for_scores(query_layer)
      key_layer = self.transpose_for_scores(self.key(hidden_states))
      value_layer = self.transpose_for_scores(self.value(hidden_states))
      attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
      attention_scores = attention_scores / math.sqrt(self.attention_head_size)

      
      # Normalize the attention scores to probabilities.
      attention_probs = self.softmax(attention_scores)

      # This is actually dropping out entire tokens to attend to, which might
      # seem a bit unusual, but is taken from the original Transformer paper.
      attention_probs = self.dropout(attention_probs)

      context_layer = torch.matmul(attention_probs, value_layer)

      context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
      new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
      context_layer = context_layer.view(*new_context_layer_shape)
      outputs = (context_layer, attention_probs) if return_att else (context_layer,)
      return outputs

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertSA(nn.Module):
    def __init__(self, 
                 drop=0.2, 
                 layer_norm=False, 
                 seq_length=128, 
                 hidden_size=64, 
                 use_company_att=False, 
                 predict_company=False,
                 company_alpha=0.2):
        super(BertSA, self).__init__()
        model_name = 'neuralmind/bert-base-portuguese-cased'
        self.config = AutoConfig.from_pretrained(model_name)  
        self.use_company_att = use_company_att
        self.predict_company = predict_company
        self.company_alpha = company_alpha

        self.bert = AutoModel.from_pretrained(model_name)
        
        if self.use_company_att:
          self.company_head = CompanyAttention(n_companies=len(COMPANIES), 
                                                dim=hidden_size, 
                                                hidden_dim=self.config.hidden_size, 
                                                dropout=drop,
                                                num_attention_heads=2,
                                                seq_length=seq_length)
          self.pooler = BertPooler(hidden_size)
        self.hidden_size = hidden_size if self.use_company_att else self.config.hidden_size
        self.classifier = ClassificationHead(self.hidden_size, drop=drop, layer_norm=layer_norm, n_classes=1)
        if self.predict_company:
          self.company_predictor = ClassificationHead(self.hidden_size, drop=drop, layer_norm=layer_norm, n_classes=len(COMPANIES))
          self.company_criterion = nn.CrossEntropyLoss()
        self.softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.sa_criterion = nn.MSELoss()#nn.CrossEntropyLoss()
        
        
    
    def embedding(self, input_ids, companies_ids):

        bert_output = self.bert(input_ids=input_ids)
        if self.use_company_att:
          x = self.company_head(bert_output[0], companies_ids, return_att=False)[0]
          x = self.pooler(x)
        else:
          x = bert_output[1]
        return x 

    def forward(self, input_ids, companies_ids, target=None):
        bert_output = self.bert(input_ids=input_ids)
        if self.use_company_att:
          x = self.company_head(bert_output[0], companies_ids, return_att=False)[0]
          x = self.pooler(x)
        else:
          x = bert_output[1]
        sa_logits = self.classifier(x)
        loss = 0
        if target is not None:
          sa_logits = self.sigmoid(sa_logits)
          loss += self.sa_criterion(sa_logits.float(), target.float())
          
        if self.predict_company:
          company_logits = self.company_predictor(x)
          company_loss = self.company_criterion(company_logits, companies_ids)
          loss += self.company_alpha * company_loss
        return (sa_logits, company_logits, loss) if self.predict_company else (sa_logits, loss)