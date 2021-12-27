import math
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset
import re

class SADataset(Dataset):

  def __init__(self, df, tokenizer, max_length, companies):
    self.df = df.copy()
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.labels = {'Neutral': 0.5, 'Positive': 1, 'Negative':0}
    self.companies = companies
    self.encode()

  def encode_txt(self, text):
    return self.tokenizer.encode(text, 
                         max_length=self.max_length, 
                         padding='max_length', 
                         truncation=True, 
                         return_tensors='pt')[0]

  def encode(self):
    self.df['input_ids'] = self.df.text.apply(lambda x: self.encode_txt(x))

  def __len__(self):
    return self.df.shape[0]
  
  def __getitem__(self, idx):
    x = self.df['input_ids'].iloc[idx]
    company_id = self.companies.index(self.df["company"].iloc[idx])
    if 'label' in self.df.columns:
      y = self.labels[self.df['label'].iloc[idx]]
      return x, company_id, y
    return x, company_id