from typing import List
import pandas as pd
import numpy as np
from csmr_kedro.models import SADataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer 
from tqdm import tqdm


def get_embedding(
    df: pd.DataFrame, 
    model: nn.Module, 
    device: str,
    max_length: int,
    batch_size: int = 64) -> pd.DataFrame:

    
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
    dataset = SADataset(df, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = np.empty((0, 768))
        Y = np.empty((0,))
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model.embedding(batch['input_ids'], None)
            preds = np.append(preds, outputs.cpu().numpy(), axis=0)
            if "label" in batch:
                Y = np.append(Y, batch['label'].cpu().numpy(), axis=0)
        X = pd.DataFrame(data=preds, columns=['feature_'+str(i+1) for i in range(preds.shape[1])])
    X.index = df.index
    X["text"] = df.text
    X["company"] = df.company
    if "tweet_id" in df.columns:
        X["tweet_id"] = df.tweet_id
    if "label" in df.columns:
        X["target"] = Y
    return X