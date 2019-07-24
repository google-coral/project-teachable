# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.classification.engine import ClassificationEngine

from pkg_resources import parse_version
from edgetpu import __version__ as edgetpu_version
assert parse_version(edgetpu_version) >= parse_version('2.11.1'), \
    'Imprinting demo requires Edge TPU version >= 2.11.1'

from edgetpu.learn.imprinting.engine import ImprintingEngine
import numpy as np
from PIL import Image


class DemoImprintingEngine(object):
  """Engine wrapping from Imprinting Engine for demo usage."""

  def __init__(self, model_path, output_path, keep_classes, batch_size):
    """Creates a ImprintingEngine with given model and labels.

    Args:
      model_path: String, path to TF-Lite Flatbuffer file.
      output_path: String, path to output tflite file.
      keep_classes: Bool, whether to keep base model classes.
      batch_size: Int, batch size for engine to train once.

    Raises:
      ValueError: An error occurred when model output is invalid.
    """
    self._model_path = model_path
    self._keep_classes = keep_classes
    self._output_path = output_path
    self._batch_size = batch_size
    self._required_image_size = self.getRequiredInputShape()
    self._example_count = 0
    self._imprinting_engine = ImprintingEngine(self._model_path, keep_classes=self._keep_classes)
    self.clear()

  def getRequiredInputShape(self):
    """
    Get the required input shape for the model.
    """
    basic_engine = BasicEngine(self._model_path)
    input_tensor_shape = basic_engine.get_input_tensor_shape()
    if (input_tensor_shape.size != 4 or input_tensor_shape[3] != 3 or
        input_tensor_shape[0] != 1):
      raise RuntimeError(
          'Invalid input tensor shape! Expected: [1, height, width, 3]')
    return (input_tensor_shape[2], input_tensor_shape[1])

  def clear(self):
    """
    Save the trained model.
    Clear the store: forgets all stored images.
    """
    # Save the trained model.
    if self._example_count > 0:
      self._imprinting_engine.SaveModel(self._output_path)
    # The size of all the image store.
    self._example_count = 0

    # The ImprintingEngine does not allow training images with too large labels.
    # For example, with an existing model with 3 output classes, there are two
    # options: training existing classes [0, 1, 2] or training exactly the next
    # class [3].

    # We have two maps to store the mappings from button_label to real_label,
    # and vice versa.
    self._label_map_button2real = {}
    self._label_map_real2button = {}
    self._max_real_label = 0
    # A map with real label as key, and training images as value.
    self._image_map = defaultdict(list)

  def trainAndUpdateModel(self):
    """Train a batch of images and update the engines."""
    for label_real in range(0, self._max_real_label):
      if label_real in self._image_map:
        self._imprinting_engine.Train(np.array(self._image_map[label_real]), label_real)
    self._image_map = defaultdict(list)  #reset

  def addImage(self, img, label_button):
    """Add an image to the store."""
    # Update the label map.
    if label_button not in self._label_map_button2real:
      self._label_map_button2real[label_button] = self._max_real_label
      self._label_map_real2button[self._max_real_label] = label_button
      self._max_real_label += 1
    label_real = self._label_map_button2real[label_button]
    self._example_count += 1
    resized_img = img.resize(self._required_image_size, Image.NEAREST)
    self._image_map[label_real].append(np.asarray(resized_img).flatten())

    # Train a batch of images.
    if sum(len(v) for v in self._image_map.values()) == self._batch_size:
      self.trainAndUpdateModel()

  def classify(self, img):
    # If we have nothing trained, the answer is None
    if self.exampleCount() == 0:
        return None
    resized_img = img.resize(self._required_image_size, Image.NEAREST)
    scores = self._imprinting_engine.ClassifyWithResizedImage(resized_img, top_k=1)
    return self._label_map_real2button[scores[0][0]]

  def exampleCount(self):
    """Just returns the size of the image store."""
    return self._example_count

