from elasticsearch import Elasticsearch
from helpers import env_helper
from typing import List

class ElasticsearchRetriever:
    def __init__(self):
        self.es = Elasticsearch(env_helper.ELASTIC_HOST)

    def search(self, index: str, query: str, total_result: int) -> List[str] :
        query_result = self.es.search(
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
        
    def search_all(self, index: str):
        res = self.es.search(
            index=index,
            query={
                "match_all": {}
            }
        )

        print(res)