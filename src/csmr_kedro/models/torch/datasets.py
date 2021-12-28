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
from csmr_kedro.extras.datasets import COMPANIES

class SADataset(Dataset):

  def __init__(self, df, tokenizer, max_length):
    self.df = df.copy()
    self.df = cleaning_data(df)
    self.tokenizer = tokenizer
    self.max_length = max_length
    self.labels = {'Neutral': 0.5, 'Positive': 1, 'Very Positive': 1, 'Negative':0, 'Very Negative': 0}
    self.companies = COMPANIES
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


def clean_tweet(text: str) -> str:
  emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)

  text = emoji_pattern.sub('', text)
  #text = text.encode('ascii', 'ignore').decode('ascii')
  links_and_replys = re.compile("(http\S+)|(https\S+)|^(@[A-Za-z0–9_]+ )+")
  text = links_and_replys.sub('', text)
  laughs = re.compile("( [kK]+ )|( (rs)+ )| (ha)+ ")
  text = laughs.sub('', text)
  hashtags = re.compile(" ([@#][A-Za-z0–9_]+ )+[@#][A-Za-z0–9_]+ *$")
  text = hashtags.sub('', text)
  return text

def cleaning_data(df: pd.DataFrame) -> pd.DataFrame:
  df["text"] = df.text.apply(clean_tweet)
  return df