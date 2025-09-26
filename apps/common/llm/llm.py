# Copyright 2024 The HAKES Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class LLM:
    """
    Base class for all LLMs (Large Language Models).
    This class defines the interface that all LLMs should implement.
    """

    def __init__(self, model_name: str, config: dict = None):
        self.model_name = model_name
        self.config = config if config is not None else {}

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text based on the provided prompt.

        :param prompt: The input text to generate a response for.
        :param max_length: The maximum length of the generated text.
        :return: Generated text as a string.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def get_model_info(self) -> dict:
        """
        Get information about the model.

        :return: A dictionary containing model information such as name, version, etc.
        """
        return {"model_name": self.model_name, "type": self.__class__.__name__}

    def __str__(self):
        """
        String representation of the LLM instance.
        :return: A string describing the LLM instance.
        """
        return f"LLM(model_name={self.model_name}, type={self.__class__.__name__})"

    def __repr__(self):
        """
        Official string representation of the LLM instance.
        :return: A string that can be used to recreate the LLM instance.
        """
        return f"{self.__class__.__name__}(model_name={self.model_name})"
