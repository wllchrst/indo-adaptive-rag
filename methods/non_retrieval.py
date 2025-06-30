from methods.base_method import BaseMethod
class NonRetrieval(BaseMethod):
    def __init__(self):
        super().__init__()

    def answer(self, query: str):
        """
        This method is a placeholder for non-retrieval methods.
        It simply returns an answer from llm without any retrieval process.
        """
        return self.llm.answer(query)