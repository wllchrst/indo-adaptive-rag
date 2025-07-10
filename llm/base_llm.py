from abc import ABC, abstractmethod
from interfaces import IDocument
class BaseLLM(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def answer(self, query: str):
        pass
    
    def format_with_document(self, prompt: str, documents: list[IDocument]) -> str:
        """
        Formats the prompt with the document content.
        """
        context_formatted = ""
        for doc in documents:
            context_formatted += f"Context Title: {doc.metadata.title}\n{doc.text}\n\n"
        
        return "\n".join([
            context_formatted,
            f"Q: {prompt}",
            "Berikan jawaban yang singkat."
        ])
    
    def validate_context(self, query: str, context:str) -> bool:
        question = f"Apakah konteks berikut cukup untuk menjawab pertanyaan: '{query}'?\n\nKonteks:\n{context}\n\nJawab dengan 'Ya' atau 'Tidak'."

        response = self.answer(question)
        print(response)
        
        return True if "Ya" in response else False