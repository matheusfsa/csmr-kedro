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

from functools import reduce
from multiprocessing import Pipe
from operator import add
from os import pipe
from xml.etree.ElementTree import PI
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import evaluate_model, get_best_model, model_choice, split_X_y
from kedro.framework.session.session import get_current_session

def split_template(name: str, has_label: bool = True) -> Pipeline:
    return Pipeline(
        [   
            
            node(
                func=split_X_y,
                inputs=["data_features"],
                outputs=["X", "y"],
                name=name
            ),
        ]
        )
def search_template(name: str) -> Pipeline:
    return Pipeline(
        [   
            
            node(
                func=get_best_model,
                inputs=["X_train",
                        "y_train",
                        "model", 
                        "params:n_iter",
                        "params:random_state"],
                outputs="best_model",
                name=name
            )
        ]
        )
def create_pipeline(**kwargs):
    try:
        session = get_current_session()
        context = session.load_context()
        catalog = context.catalog

        models = catalog.load("params:models")
    except:
        models = ["random_forest", "ada", "xgb"]
    data_refs = ["train", "test"]

    split_pipelines = [
        pipeline(
            pipe=split_template(f"split_{ref}"),
            inputs={"data_features": f"{ref}_features"},
            outputs={"X": f"X_{ref}",
                     "y": f"y_{ref}"}
        )
        for ref in data_refs
    ]
    split_pipeline = reduce(add, split_pipelines)
    search_pipelines = [
        pipeline(
            pipe=search_template(f"get_best_{model}"),
            inputs={"model": f"params:models.{model}"},
            outputs={"best_model": f"best_{model}"}
        )
        for model in models
    ]
    search_pipeline = reduce(add, search_pipelines)

    choice_pipeline = Pipeline(
        [
            node(
                func=model_choice,
                inputs=["X_train", "y_train"] + [f"best_{model}" for model in models],
                outputs="model",
                name="model_choice"
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test"],
                outputs="model_metrics",
                name="evaluate_model"
            )
        ]
    )
    return split_pipeline + search_pipeline + choice_pipeline
