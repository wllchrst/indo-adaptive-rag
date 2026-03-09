import httpx
from llm.base_llm import BaseLLM
from helpers import env_helper
from ollama import Client

OLLAMA_MODEL_LIST = ['deepseek-r1:latest', 'bangundwir/bahasa-4b-chat', 'gemma3:latest', 'qwen3:8b']
DEFAULT_TIMEOUT = 180
DEFAULT_OPTIONS = {
    "temperature": 0.1,
    "top_k": 5,
    "top_p": 0.1,
    "repeat_penalty": 1.0,
    "seed": 42
}


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model_name='bangundwir/bahasa-4b-chat',
        timeout: int = DEFAULT_TIMEOUT,
        default_options: dict = None
    ):
        super().__init__()
        self.API_KEY = env_helper.GEMINI_API_KEY
        self.client = Client(host=env_helper.OLLAMA_HOST, timeout=timeout)
        self.model_name = model_name
        self.timeout = timeout
        self.default_options = default_options or DEFAULT_OPTIONS.copy()

    def answer(self, prompt: str, options: dict = None) -> str:
        try:
            final_options = {**self.default_options, **(options or {})}

            response = self.client.chat(
                self.model_name,
                think=False,
                stream=False,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                options=final_options
            )

            return response.message.content
        except httpx.ReadTimeout as timeout:
            raise TimeoutError(
                f'Ollama request exceeded timeout limit for model {self.model_name} with error: {timeout}')
