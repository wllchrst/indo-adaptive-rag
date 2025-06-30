from abc import ABC, abstractmethod
from helpers import env_helper
class BaseLLM(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def answer(self, query: str):
        pass