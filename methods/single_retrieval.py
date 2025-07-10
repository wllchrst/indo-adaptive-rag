from methods.base_method import BaseMethod
from helpers import WordHelper

class SingleRetrieval(BaseMethod):
    def __init__(self, model_type: str):
        super().__init__(model_type)
    
    def answer(self,
               query: str,
               with_logging: bool = False, 
               index: str = ''):
        """
        This method retrieves a single relevant document from the vector database
        and uses it to answer the query.
        """
        print(f"Index: {index}")
        retrieval_query = WordHelper.remove_stop_words(query)
        retrieved_document = self.retrieve_document(
            query=retrieval_query,
            index=index
        )

        formatted_query = self.llm.format_with_document(query, retrieved_document)
        answer = self.llm.answer(formatted_query)

        self.log_actions(
            method="Single Retrieval",
            query=formatted_query,
            answer=answer,
            with_logging=with_logging
        )

        return answer