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

from .llm import LLM
import openai


class HealthGPT(LLM):
    """
    HealthGPT LLM implementation.
    This class extends the base LLM class and provides specific functionality for HealthGPT models.
    """

    def __init__(self, model_name: str, config: dict = None):
        super().__init__(model_name, config)
        if config is None:
            config = {}
        self.api_key = config.get("api_key", None)
        self.base_url = config.get("base_url", None)
        self.llm = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        # track conversation state
        self.last_response_id = None

    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text based on the provided prompt.

        :param prompt: The input text to generate a response for.
        :param max_length: The maximum length of the generated text.
        :return: Generated text as a string.
        """
        response = self.llm.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and harmless assistant. You should think step-by-step. You should provide consise and accurate answers based on the context provided.",
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            model=self.model_name,
            stream=False,
            max_tokens=max_length,
            temperature=0.6,
            top_p=0.7,
            extra_body={
                "top_k": 50,
            },
            frequency_penalty=0,
        )
        self.last_response_id = response.id
        return response.choices[0].message.content
