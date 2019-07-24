#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Currently, the edgetpu api for imprinting has not been updated.
# You might need to clone the python-tflite-source and checkout to
# imprinting-improvement branch, and set the path below:
python3 teachable.py --model 'models/mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite' --method 'imprinting'
