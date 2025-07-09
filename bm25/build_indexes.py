import pandas as pd
import traceback
from elasticsearch import Elasticsearch
from typing import List, TypedDict
from helpers import env_helper

# Make sure you have run the docker compose file
es = Elasticsearch(env_helper.ELASTIC_HOST)

class Document(TypedDict):
    id: str
    text: str
    question: str
    answer: str

def make_indoqa_context() -> List[Document]:
    from classification import gather_indo_qa
    train_df, test_df = gather_indo_qa(remove_column=False)
    full_df = pd.concat([train_df, test_df])
    full_df = full_df.drop_duplicates(subset=['context'])

    docs: List[Document] = []
    for index, row in full_df.iterrows():
        doc: Document = {
            'id':row['id'],
            'answer': row['answer'],
            'question': row['question'],
            'text': row['context']
        }

        docs.append(doc)

    return docs

def insert_documents(index: str, documents: List[Document]):
    operations = []
    for document in documents:
        operations.append({'index': {'_index': index}})
        operations.append(document)
        
    es.bulk(operations=operations)

def build_all_index():
    try:
        print(f"Elasticsearch information: {es.info()}")
        # index names
        indoqa_index = 'indoqa'

        # INDOQA
        indoqa_docs = make_indoqa_context()
        es.indices.delete(index=indoqa_index)
        es.indices.create(index=indoqa_index)
        insert_documents(indoqa_index, indoqa_docs)
    except Exception as e:
        print(f"Error while building all index: {e}")
        traceback.print_exc()