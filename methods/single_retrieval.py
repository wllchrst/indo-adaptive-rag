from methods.base_method import BaseMethod
from helpers import WordHelper
from typing import Optional, Tuple


class SingleRetrieval(BaseMethod):
    def __init__(self, model_type: str):
        super().__init__(model_type)

    def answer(self,
               query: str,
               with_logging: bool = False,
               index: str = '',
               answer: Optional[str] = None,
               supporting_facts: list[str] = []) -> Tuple[str, int]:
        """
        This method retrieves a single relevant document from the vector database
        and uses it to answer the query.
        """
        retrieval_query = WordHelper.clean_sentence(WordHelper.remove_stop_words(query))

        retrieved_document = self.retrieve_document(
            query=retrieval_query,
            index=index
        )

        formatted_query = self.llm.format_with_document(query, retrieved_document)
        prediction = self.llm.answer(formatted_query)

        self.log_actions(
            method="Single Retrieval",
            query=formatted_query,
            answer=prediction,
            with_logging=with_logging
        )

        return prediction.strip(), 1
