from methods.base_method import BaseMethod
from typing import Optional

class NonRetrieval(BaseMethod):
    def __init__(self, model_type: str):
        super().__init__(model_type)

    def answer(self,
               query: str,
               with_logging: bool = False,
               index: str = '',
               answer: Optional[str] = None):
        """True
        This method is a placeholder for non-retrieval methods.
        It simply returns an answer from llm without any retrieval process.
        """
        format_query = f"Q: {query}\nBerikan jawaban yang singkat."
        prediction = self.llm.answer(format_query)

        self.log_actions(
            method="Non Retrieval",
            query=format_query,
            answer=prediction,
            with_logging=with_logging
        )

        return prediction.strip()