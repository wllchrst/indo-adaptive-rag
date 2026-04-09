from elasticsearch import Elasticsearch
from helpers import env_helper
from joblib import Memory

memory = Memory(location=env_helper.CACHE_DIRECTORY, verbose=0)
es = Elasticsearch(env_helper.ELASTIC_HOST)

EMBEDDING_FIELD = 'text_embedding'
EMBEDDING_DIMENSION = 384
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


class ElasticsearchRetriever:
    @staticmethod
    def _get_embedding(query: str) -> list[float]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return model.encode(query, normalize_embeddings=True).tolist()

    @staticmethod
    def search(index: str, query: str, total_result: int):
        print("Searching not from cache")
        query_result = es.search(
            index=index,
            size=total_result,
            query={
                'match': {
                    'text': query
                }
            }
        )

        hits = query_result['hits']['hits']

        return hits

    @staticmethod
    def search_knn(index: str, query: str, total_result: int,
                   knn_field: str = EMBEDDING_FIELD, num_candidates: int = 100):
        print("Searching with knn")
        query_vector = ElasticsearchRetriever._get_embedding(query)

        query_result = es.search(
            index=index,
            size=total_result,
            knn={
                'field': knn_field,
                'query_vector': query_vector,
                'k': total_result,
                'num_candidates': num_candidates,
            }
        )

        return query_result['hits']['hits']

    @staticmethod
    def search_hybrid(index: str, query: str, total_result: int,
                      knn_field: str = EMBEDDING_FIELD, num_candidates: int = 100):
        print("Searching with hybrid (BM25 + knn)")
        query_vector = ElasticsearchRetriever._get_embedding(query)

        query_result = es.search(
            index=index,
            size=total_result,
            query={
                'match': {
                    'text': query
                }
            },
            knn={
                'field': knn_field,
                'query_vector': query_vector,
                'k': total_result,
                'num_candidates': num_candidates,
            }
        )

        return query_result['hits']['hits']

    @staticmethod
    def ensure_embedding_mapping(index: str, knn_field: str = EMBEDDING_FIELD,
                                 dimension: int = EMBEDDING_DIMENSION):
        mapping = es.indices.get_mapping(index=index)
        properties = mapping[index]['mappings'].get('properties', {})
        if knn_field not in properties:
            es.indices.put_mapping(
                index=index,
                properties={
                    knn_field: {
                        'type': 'dense_vector',
                        'dims': dimension,
                        'index': True,
                        'similarity': 'cosine',
                    }
                }
            )
            print(f"Added {knn_field} mapping to index {index}")
        else:
            print(f"Index {index} already has {knn_field} mapping")

    @staticmethod
    def search_all(index: str):
        res = es.search(
            index=index,
            query={
                "match_all": {}
            }
        )

        print(res)
