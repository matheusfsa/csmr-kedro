import pandas as pd
from csmr_kedro.extras.datasets import COMPANIES
from csmr_kedro.models import get_embedding
import torch.nn as nn

def drop_invalid_samples(df: pd.DataFrame):
    df = df[df.company.isin(COMPANIES)]
    df = df[~df.text.isna()]
    return df

def feature_extraction(
    df: pd.DataFrame, 
    model: nn.Module, 
    device: str,
    max_length: int,
    batch_size: int = 64):
    
    df = get_embedding(df, model, device, max_length, batch_size)
    return df 