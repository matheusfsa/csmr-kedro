import pandas as pd
from csmr_kedro.extras.datasets import COMPANIES

def preprocess(data: pd.DataFrame):
    
    for c in COMPANIES:
        data[f"company_{c}"] = (data.company == c) * 1
    X = data.drop(columns=["text","company"])
    if "target" in data.columns:
        y = data.target
        data.drop(columns=["target"])
        return X, y
    return X
