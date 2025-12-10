# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# specifies the available options for score detection

NETWORK_MODELS = {"crnn_ctc"}
DIRECTED_MODELS = {}

ALL_MODELS = sorted(NETWORK_MODELS.union(DIRECTED_MODELS))
DEFAULT_MODEL = "crnn_ctc"


def get_model_type(name: str) -> str:
    name = name.lower()
    if name in NETWORK_MODELS:
        return "network"
    elif name in DIRECTED_MODELS:
        return "directed"
    else:
        raise ValueError(f"Invalid model name: {name}")
