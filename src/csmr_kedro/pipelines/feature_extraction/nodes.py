# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is a boilerplate pipeline 'feature_extraction'
generated using Kedro 0.17.4
"""
from typing import List
import pandas as pd
import numpy as np
from csmr_kedro.models import SADataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer 

def get_embedding(
    df: pd.DataFrame, 
    model: nn.Module, 
    device: str,
    max_length: int) -> pd.DataFrame:

    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
    dataset = SADataset(df, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        preds = np.empty((0, 768))
        Y = np.empty((0,))
        for i, (x, companies_ids, y) in enumerate(dataloader):
            x = x.to(device)
            companies_ids = companies_ids.to(device)
            y = y.to(device)
            outputs = model.embedding(x, None)
            preds = np.append(preds, outputs.cpu().numpy(), axis=0)
            Y = np.append(Y, y.cpu().numpy(), axis=0)
            X = pd.DataFrame(data=preds, columns=['f'+str(i+1) for i in range(preds.shape[1])])
    X.index = df.index
    X["text"] = df.text
    X["company"] = df.company
    X["target"] = Y
    return X
