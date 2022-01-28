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
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.17.5
"""
from typing import Dict, List, Tuple, Union
from sklearn.model_selection import KFold, RandomizedSearchCV
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, f1_score, mean_squared_error 
from csmr_kedro.extras.helpers import model_from_string, load_params
from csmr_kedro.extras.datasets import COMPANIES
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from csmr_kedro.extras.helpers import preprocess
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline

def split_X_y(
    data: pd.DataFrame, 
    label: str = "target") -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """
    This node split data to be used in training
    
    Parameters:
    data (pd.DataFrame): Dataframe with features (and labels).
    label (str): Label column name

    Return:
    Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]: Splited data
    """
    if label in data.columns:
        return data.drop(columns=["text", label]), data[label]
    return data.drop(columns=["text"])
    

def build_model(
    X_train: pd.DataFrame, 
    model: Dict):
    
    estimator = model_from_string(model["model_class"], model["default_params"])

    numeric_features = list(filter(lambda x: x.startswith("feature_"), X_train.columns))
    categorical_features = ["company"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", 'passthrough', numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    

    return preprocessor, estimator

def get_best_model(
    X: pd.DataFrame, 
    y: pd.Series,
    model: Dict, 
    random_state: int) -> Tuple[BaseEstimator, float]:
    
    preprocessor, estimator = build_model(X, model)
    grid = load_params(model["params"])
    cv = KFold(n_splits=2)
    scorer = make_scorer(accuracy_score)
    search = RandomizedSearchCV(estimator, grid, cv=cv,scoring=scorer, verbose=2, random_state=random_state, n_iter=1, n_jobs=-1)
    X = preprocessor.fit_transform(X)
    search.fit(X, y.values)
    pipe = Pipeline(steps=[
        ('data_transformer', preprocessor),
        ('estimator', search.best_estimator_)
    ])
    return pipe, search.best_score_

def model_choice(
    X: pd.DataFrame, 
    y: pd.Series,
    *models):

    models = list(models)
    models.sort(key=lambda tup: tup[1])
    model = models[0][0]
    return model.fit(X, y.values)

def evaluate_model(
    model: BaseEstimator, 
    X_test: pd.DataFrame,
    y_true: pd.Series) -> Dict[str, Union[float, List[float]]]:

    y_pred = model.predict(X_test)
    return {
        "accuracy": {"value": accuracy_score(y_true, y_pred), "step":1},
        "precision": {"value": precision_score(y_true, y_pred, average='weighted'), "step":1},
        "recall": {"value": recall_score(y_true, y_pred, average='weighted'), "step":1},
        "f1": {"value": f1_score(y_true, y_pred, average='weighted'), "step":1},
        }