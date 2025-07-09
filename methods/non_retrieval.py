from methods.base_method import BaseMethod
class NonRetrieval(BaseMethod):
    def __init__(self):
        super().__init__()

    def answer(self, query: str, with_logging: bool = False, index: str = ''):
        """True
        This method is a placeholder for non-retrieval methods.
        It simply returns an answer from llm without any retrieval process.
        """
        format_query = f"Q: {query}\nBerikan jawaban yang singkat."
        answer = self.llm.answer(format_query)

        self.log_actions(
            method="Non Retrieval",
            query=format_query,
            answer=answer,
            with_logging=with_logging
        )

        return answer