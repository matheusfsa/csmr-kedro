import pandas as pd
from csmr_kedro.extras.datasets import COMPANIES


def drop_invalid_samples(df: pd.DataFrame):
    df = df[df.company.isin(COMPANIES)]
    df = df[~df.text.isna()]
    return df