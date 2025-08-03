from llm.base_llm import BaseLLM
from helpers import env_helper
from ollama import Client

OLLAMA_MODEL_LIST = ['deepseek-r1:latest', 'bangundwir/bahasa-4b-chat']

class OllamaLLM (BaseLLM):
    def __init__(self, model_name = 'bangundwir/bahasa-4b-chat'):
        super().__init__()
        self.API_KEY = env_helper.GEMINI_API_KEY
        self.client = Client(host=env_helper.OLLAMA_HOST)
        self.model_name = model_name

    def answer(self, prompt: str) -> str:
        response = self.client.chat(self.model_name, think=False, stream=False, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])

        return response.message.content
