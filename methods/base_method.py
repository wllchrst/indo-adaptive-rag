from abc import ABC, abstractmethod
from llm import GeminiLLM, HuggingFaceLLM, OllamaLLM, OLLAMA_MODEL_LIST
# from vector_database import DatabaseHandler
from interfaces import IDocument, IMetadata
from bm25 import ElasticsearchRetriever
from typing import List, Optional, Tuple, Dict, Literal
from llm.ollama_llm import OLLAMA_MODEL_LIST
import os
import pandas as pd

model_type_list = ['gemini', 'hugging_face']
RetrievalType = Literal['lexical', 'semantic', 'hybrid']


class BaseMethod(ABC):
    def __init__(self, model_type='gemini'):
        super().__init__()
        self.assign_llm(model_type=model_type)
        # self.database_handler = DatabaseHandler()
        self.elastic_retriever = ElasticsearchRetriever()
        self.mappings: Dict[str, Dict[str, List[str]]] = {}
        self.load_all_mappings()

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
               supporting_facts: list[str] = [], question_id: Optional[str] = None,
               retrieval_type: RetrievalType = 'lexical') -> Tuple[str, int, Optional[Dict]]:
        pass

    def retrieve_document(self,
                          query: str,
                          total_result: int = 3,
                          index: str = '',
                          retrieval_type: RetrievalType = 'lexical') -> List[IDocument]:
        """
        Retrieve documents based on the specified retrieval type.
        
        Args:
            query: The search query
            total_result: Number of documents to retrieve
            index: Elasticsearch index name
            retrieval_type: 'lexical' (BM25), 'semantic' (knn), or 'hybrid' (RRF)
        """
        use_chromadb = True if index == '' else False

        if use_chromadb:
            return self.retrieve_chromadb(query, total_result)

        if retrieval_type == 'semantic':
            return self.retrieve_semantic(query, total_result, index)
        elif retrieval_type == 'hybrid':
            return self.retrieve_hybrid(query, total_result, index)
        else:
            return self.retrieve_elasticsearch(query, total_result, index)

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

    def retrieve_semantic(self, query: str, total_result: int, index: str) -> List[IDocument]:
        search_result = self.elastic_retriever.search_knn(index=index, query=query, total_result=total_result)

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

    def retrieve_hybrid(self, query: str, total_result: int, index: str,
                        rrf_k: int = 60, lexical_weight: float = 0.5) -> List[IDocument]:
        lexical_docs = self.retrieve_elasticsearch(query, total_result * 2, index)
        semantic_docs = self.retrieve_semantic(query, total_result * 2, index)

        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, IDocument] = {}

        for rank, doc in enumerate(lexical_docs):
            doc_id = doc.metadata.docid
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + lexical_weight / (rrf_k + rank + 1)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(semantic_docs):
            doc_id = doc.metadata.docid
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + (1 - lexical_weight) / (rrf_k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        merged_docs = []
        for doc_id in sorted_ids[:total_result]:
            merged_docs.append(doc_map[doc_id])

        return merged_docs

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

    def load_all_mappings(self):
        indices = ['indoqa', 'hotpot', 'qasina']
        for index in indices:
            self.mappings[index] = self.load_single_mapping(index)

    def load_single_mapping(self, index: str) -> Dict[str, List[str]]:
        mapping_file = f'mappings/{index}_mapping.csv'
        if not os.path.exists(mapping_file):
            return {}
        
        df = pd.read_csv(mapping_file)
        mapping = {}
        for _, row in df.iterrows():
            mapping[row['question_id']] = row['context_ids'].split(',')
        return mapping

    def calculate_hit_rate(self, retrieved_docs: List[IDocument], expected_ids: List[str]) -> Dict[str, any]:
        retrieved_ids = [doc.metadata.docid for doc in retrieved_docs]
        hits = [rid for rid in retrieved_ids if rid in expected_ids]
        
        return {
            'hits': len(hits),
            'total_retrieved': len(retrieved_docs),
            'expected_count': len(expected_ids),
            'hit_rate': len(hits) / max(len(expected_ids), 1) if expected_ids else 0,
            'retrieved_ids': retrieved_ids,
            'expected_ids': expected_ids
        }
