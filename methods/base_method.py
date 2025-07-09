from nltk.tokenize import word_tokenize
from abc import ABC, abstractmethod
from llm import GeminiLLM
from vector_database import DatabaseHandler
from interfaces import IDocument, IMetadata
from bm25 import ElasticsearchRetriever
from typing import List

class BaseMethod(ABC):
    def __init__(self):
        super().__init__()
        self.llm = GeminiLLM()
        self.database_handler = DatabaseHandler()
        self.elastic_retriever = ElasticsearchRetriever()
    
    @abstractmethod
    def answer(self, query: str, with_logging: bool, index: str):
        pass

    def retrieve_document(self,
                            query: str,
                            total_result: int = 5,
                            index: str = '') -> IDocument:
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
        return documents[0]
    
    def retrieve_elasticsearch(self, query: str, total_result: int, index: str):
        search_result = self.elastic_retriever.search(index=index, query=query, total_result=total_result)

        documents: List[IDocument] = []
        for result in search_result:
            source = result['_source']
            documents.append(IDocument(
                text=source['text'],
                distance=result['_score'],
                metadata=IMetadata(
                    docid= result['_id'],
                    source= result['_index'],
                    title=''
                )
            ))
        
        return documents
    
    def retrieve_chromadb(self, query: str, total_result: int = 5):
        collections = self.database_handler.get_collections()
        document = None

        for collection in collections:
            result = self.database_handler.query(
                collection_name=collection.name,
                query=query,
                total_result=total_result
            )

            if len(result['documents'][0]) > 0:
                current_document = IDocument(
                    text=result['documents'][0][0],
                    distance=result['distances'][0][0],
                    metadata=IMetadata(**result['metadatas'][0][0])
                )

                if document is None or current_document['distance'] < document['distance']:
                    document = current_document

        return document
    
    def retrieve_documents(self, query: str, total_result: int = 5) -> list[str]:
        """
        This method retrieves multiple documents from the vector database based on the query.
        """
        collections = self.database_handler.get_collections()
        documents = []

        for collection in collections:
            result = self.database_handler.query(
                collection_name=collection.name,
                query=query,
                total_result=total_result
            )

            if len(result['documents'][0]) > 0:
                texts = result['documents'][0]
                distances = result['distances'][0]
                metadatas = result['metadatas'][0]

                for text, distance, metadata in zip(texts, distances, metadatas):
                    formatted_document = {
                        "text": text,
                        "distance": distance,
                        "metadata": metadata
                    }

                    documents.append(formatted_document)

        documents.sort(key=lambda doc: doc["distance"])
        return documents[:5]
    
    def log_actions(self, method: str, query: str, answer: str, with_logging: bool):
        if not with_logging:
            return
        
        print("*" * 40)
        print(method)
        print(f'Query: {query}')
        print(f'Answer: {answer}')