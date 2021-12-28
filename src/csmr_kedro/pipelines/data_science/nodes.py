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
from typing import Dict, List, Union
from sklearn.model_selection import KFold, RandomizedSearchCV
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, f1_score, mean_squared_error 
from csmr_kedro.extras.helpers import model_from_string, load_params
from csmr_kedro.extras.datasets import COMPANIES

def preprocess(data: pd.DataFrame):
    y = data.target
    for c in COMPANIES:
        data[f"company_{c}"] = (data.company == c) * 1
    X = data.drop(columns=["text","company", "target"])
    return X, y

def get_best_model(
    train_data: pd.DataFrame, 
    model: BaseEstimator, 
    random_state: int):
    X, y = preprocess(train_data)
    grid = load_params(model["params"])
    estimator = model_from_string(model["model_class"], model["default_params"])
    cv = KFold(n_splits=2)
    scorer = make_scorer(mean_squared_error)
    search = RandomizedSearchCV(estimator, grid, cv=cv, n_jobs=None,scoring=scorer, verbose=2, random_state=random_state, n_iter=1)
    search.fit(X.values, y.values)
    return search.best_estimator_, search.best_score_

def model_choice(
    train_data: pd.DataFrame, 
    *models):
    X, y = preprocess(train_data)
    models = list(models)
    models.sort(key=lambda tup: tup[1])
    model = models[0][0]
    return model.fit(X, y)

def evaluate_model(
    model: BaseEstimator, 
    test: pd.DataFrame) -> Dict[str, Union[float, List[float]]]:
    X_test, y_test = preprocess(test)
    y_pred = model.predict(X_test)
    y_pred[(y_pred > 0.3) & (y_pred < 0.7)] = 0.5
    y_pred[y_pred >= 0.7] = 1.0
    y_pred[y_pred <= 0.3] = 0.0
    accuracy = (y_pred == y_test).sum()/y_test.shape[0]
    return {
        "accuracy": {"value": accuracy, "step":1}
        }