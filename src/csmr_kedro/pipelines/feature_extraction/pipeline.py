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

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import get_embedding
from functools import reduce
from operator import add

def feature_extraction_template(name:str) -> Pipeline:
    return Pipeline(
        [
            node(func=get_embedding,
                inputs=[
                        "data", 
                        "torch_model", 
                        "params:device", 
                        "params:max_length"],
                outputs="data_features",
                name=f"{name}_feature_extraction"),
        ]
    )

def create_pipeline(**kwargs):
    refs = ["train", "test"]

    feature_extraction_pipelines = [
        pipeline(
            pipe=feature_extraction_template(data_ref),
            parameters={},
            inputs={"data": data_ref},
            outputs={"data_features": f"{data_ref}_features"}
        )
        for data_ref in refs
    ]
    all_pipelines = reduce(add, feature_extraction_pipelines)
    return all_pipelines
