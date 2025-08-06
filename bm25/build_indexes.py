import pandas as pd
import os
import traceback
import ast
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
    for _, row in full_df.iterrows():
        doc: Document = {
            'id':row['id'],
            'answer': row['answer'],
            'question': row['question'],
            'text': row['context']
        }

        docs.append(doc)

    return docs

def make_musique_context(path: str):
    file_names = os.listdir(path)
    docs: List[Document] = []

    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            contexts = row['contexts']
            context = ast.literal_eval(contexts)[0]
            for sentence in context['sentences']:
                doc: Document = {
                    'id':row['id'],
                    'answer': row['answer'],
                    'question': row['question'],
                    'text': sentence
                }

                docs.append(doc)

    return docs

def check_index_exists(index_name: str) -> bool:
    return es.indices.exists(index=index_name)
def insert_documents(index: str, documents: List[Document]):
    operations = []
    for document in documents:
        operations.append({'index': {'_index': index}})
        operations.append(document)
        
    es.bulk(operations=operations)

def build_all_index():
    try:
        print(f"Elasticsearch information: {es.info()}")
        indoqa_index = 'indoqa'
        musique_index = 'musique'

        # INDOQA
        if not check_index_exists(indoqa_index):
            print("Inserting indoqa context")
            indoqa_docs = make_indoqa_context()
            es.indices.delete(index=indoqa_index, ignore_unavailable=True)
            es.indices.create(index=indoqa_index)
            insert_documents(indoqa_index, indoqa_docs)
        else:
            print("Indoqa index already exists")


        if not check_index_exists(musique_index):
            print("Inserting musique dataset context")
            musique_docs = make_musique_context("musique")
            es.indices.delete(index=musique_index, ignore_unavailable=True)
            es.indices.create(index=musique_index)
            insert_documents(musique_index, musique_docs)
        else:
            print("Musique index already exists")

    except Exception as e:
        print(f"Error while building all index: {e}")
        traceback.print_exc()