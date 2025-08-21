from abc import ABC, abstractmethod
from llm import GeminiLLM, HuggingFaceLLM, OllamaLLM, OLLAMA_MODEL_LIST
# from vector_database import DatabaseHandler
from interfaces import IDocument, IMetadata
from bm25 import ElasticsearchRetriever
from typing import List, Optional, Tuple
from llm.ollama_llm import OLLAMA_MODEL_LIST

model_type_list = ['gemini', 'hugging_face']


class BaseMethod(ABC):
    def __init__(self, model_type='gemini'):
        super().__init__()
        self.assign_llm(model_type=model_type)
        # self.database_handler = DatabaseHandler()
        self.elastic_retriever = ElasticsearchRetriever()

    def assign_llm(self, model_type: str):
        if model_type not in model_type_list and model_type not in OLLAMA_MODEL_LIST:
            raise ValueError(f'Model type must be in this list {model_type_list} | {OLLAMA_MODEL_LIST}')

        print(f'Using model type {model_type}')
        if model_type == 'gemini':
            self.llm = GeminiLLM()
        elif model_type == 'hugging_face':
            self.llm = HuggingFaceLLM()
        else:
            self.llm = OllamaLLM(model_name=model_type)

    @abstractmethod
    def answer(self, query: str, with_logging: bool, index: str, answer: Optional[str] = None,
               supporting_facts: list[str] = []) -> Tuple[str, int]:
        pass

    def retrieve_document(self,
                          query: str,
                          total_result: int = 3,
                          index: str = '') -> List[IDocument]:
        """
        This method retrieves a single document from the vector database based on the query.
        """
        use_chromadb = True if index == '' else False

        if use_chromadb:
            return self.retrieve_chromadb(query, total_result)

        documents = self.retrieve_elasticsearch(
            query=query,
            total_result=total_result,
            index=index
        )
        return documents

    def retrieve_elasticsearch(self, query: str, total_result: int, index: str) -> List[IDocument]:
        search_result = self.elastic_retriever.search(index=index, query=query, total_result=total_result)

        documents: List[IDocument] = []
        for result in search_result:
            source = result['_source']
            documents.append(IDocument(
                text=source['text'],
                distance=result['_score'],
                metadata=IMetadata(
                    docid=result['_id'],
                    source=result['_index'],
                    title=''
                )
            ))

        return documents

    def retrieve_chromadb(self, query: str, total_result: int = 5) -> List[IDocument]:
        return []
        # collections = self.database_handler.get_collections()
        #
        # for collection in collections:
        #     result = self.database_handler.query(
        #         collection_name=collection.name,
        #         query=query,
        #         total_result=total_result
        #     )
        #
        #     documents: List[IDocument] = []
        #
        #     if len(result['documents'][0]) > 0:
        #         current_document = IDocument(
        #             text=result['documents'][0][0],
        #             distance=result['distances'][0][0],
        #             metadata=IMetadata(**result['metadatas'][0][0])
        #         )
        #
        #         documents.append(current_document)
        #
        # return documents

    def log_actions(self, method: str, query: str, answer: str, with_logging: bool):
        if not with_logging:
            return

        print("*" * 40)
        print(method)
        print(f'Query: {query}')
        print(f'Answer: {answer}')
