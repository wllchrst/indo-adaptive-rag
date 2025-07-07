from methods.base_method import BaseMethod
from helpers import WordHelper

class SingleRetrieval(BaseMethod):
    def __init__(self):
        super().__init__()
    
    def answer(self, query: str, with_logging: bool = False):
        """
        This method retrieves a single relevant document from the vector database
        and uses it to answer the query.
        """
        retrieval_query = WordHelper.remove_stop_words(query)
        retrieved_document = self.retrieve_document(retrieval_query)

        formatted_query = self.llm.format_with_document(query, [retrieved_document])
        answer = self.llm.answer(formatted_query)

        self.log_actions(
            method="Single Retrieval",
            query=formatted_query,
            answer=answer,
            with_logging=with_logging
        )

        return answer