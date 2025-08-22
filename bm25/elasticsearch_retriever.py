from elasticsearch import Elasticsearch
from helpers import env_helper
from joblib import Memory
from helpers import env_helper

memory = Memory(location=env_helper.CACHE_DIRECTORY, verbose=0)
es = Elasticsearch(env_helper.ELASTIC_HOST)

class ElasticsearchRetriever:
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
    def search_all(index: str):
        res = es.search(
            index=index,
            query={
                "match_all": {}
            }
        )

        print(res)
