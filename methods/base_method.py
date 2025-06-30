from abc import ABC, abstractmethod
from llm import GeminiLLM

class BaseMethod(ABC):
    def __init__(self):
        super().__init__()
        self.llm = GeminiLLM()
    
    @abstractmethod
    def answer(self, query: str):
        pass