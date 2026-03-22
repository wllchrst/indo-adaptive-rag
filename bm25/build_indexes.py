import pandas as pd
import os
import traceback
import ast
import uuid
from elasticsearch import Elasticsearch
from typing import List, TypedDict, Dict
from helpers import env_helper

# Make sure you have run the docker compose file
es = Elasticsearch(env_helper.ELASTIC_HOST)


class Document(TypedDict):
    id: str
    text: str
    question: str
    answer: str


def make_indoqa_context() -> tuple[List[Document], Dict[str, List[str]]]:
    from classification import gather_indo_qa
    train_df, test_df = gather_indo_qa()
    full_df = pd.concat([train_df, test_df])

    docs: List[Document] = []
    mapping: Dict[str, List[str]] = {}

    for _, row in full_df.iterrows():
        question_id = str(row['id'])
        context_id = str(uuid.uuid4())
        
        doc: Document = {
            'id': context_id,
            'answer': row['answer'],
            'question': row['question'],
            'text': row['context']
        }

        docs.append(doc)
        
        if question_id not in mapping:
            mapping[question_id] = []
        mapping[question_id].append(context_id)

    return docs, mapping


def make_hotpot_context(path: str) -> tuple[List[Document], Dict[str, List[str]]]:
    file_names = os.listdir(path)
    docs: List[Document] = []
    mapping: Dict[str, List[str]] = {}
    
    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        df = pd.read_csv(file_path)

        for _, row in df.iterrows():
            try:
                contexts = row['contexts']
                literal_eval = ast.literal_eval(contexts)
                if len(literal_eval) == 0:
                    continue

                context = literal_eval[0]
                question_id = str(row['id'])
                
                if question_id not in mapping:
                    mapping[question_id] = []
                
                for sentence in context['sentences']:
                    context_id = str(uuid.uuid4())
                    
                    doc: Document = {
                        'id': context_id,
                        'answer': row['answer'],
                        'question': row['question'],
                        'text': sentence
                    }

                    docs.append(doc)
                    mapping[question_id].append(context_id)
            except Exception as exception:
                raise exception

    return docs, mapping


def make_qasina_context() -> tuple[List[Document], Dict[str, List[str]]]:
    from classification import gather_qasina_data
    df = gather_qasina_data()

    docs: List[Document] = []
    mapping: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        question_id = str(row['ID'])
        context_id = str(uuid.uuid4())
        
        doc: Document = {
            'id': context_id,
            'answer': row['answer'],
            'question': row['question'],
            'text': row['context']
        }

        docs.append(doc)
        
        if question_id not in mapping:
            mapping[question_id] = []
        mapping[question_id].append(context_id)

    return docs, mapping


def check_index_exists(index_name: str) -> bool:
    return es.indices.exists(index=index_name)


def insert_documents(index: str, documents: List[Document]):
    operations = []
    for document in documents:
        operations.append({'index': {'_index': index, '_id': document['id']}})
        operations.append(document)

    es.bulk(operations=operations)


def save_mapping_to_csv(mapping: Dict[str, List[str]], filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    mapping_data = []
    for question_id, context_ids in mapping.items():
        mapping_data.append({
            'question_id': question_id,
            'context_ids': ','.join(context_ids)
        })
    
    df = pd.DataFrame(mapping_data)
    df.to_csv(filename, index=False)


def build_all_index():
    try:
        print(f"Elasticsearch information: {es.info()}")
        indoqa_index = 'indoqa'
        hotpot_index = 'hotpot'
        qasina_index = 'qasina'

        # INDOQA
        if not check_index_exists(indoqa_index):
            print("Inserting indoqa context")
            indoqa_docs, indoqa_mapping = make_indoqa_context()
            es.indices.delete(index=indoqa_index, ignore_unavailable=True)
            es.indices.create(index=indoqa_index)
            insert_documents(indoqa_index, indoqa_docs)
            save_mapping_to_csv(indoqa_mapping, 'mappings/indoqa_mapping.csv')
        else:
            print("Indoqa index already exists")

        if not check_index_exists(hotpot_index):
            print("Inserting hotpot dataset context")
            hotpot_docs, hotpot_mapping = make_hotpot_context("hotpot")
            es.indices.delete(index=hotpot_index, ignore_unavailable=True)
            es.indices.create(index=hotpot_index)
            insert_documents(hotpot_index, hotpot_docs)
            save_mapping_to_csv(hotpot_mapping, 'mappings/hotpot_mapping.csv')
        else:
            print("hotpot index already exists")

        if not check_index_exists(qasina_index):
            print("Inserting qasina dataset context")
            qasina_docs, qasina_mapping = make_qasina_context()
            es.indices.delete(index=qasina_index, ignore_unavailable=True)
            es.indices.create(index=qasina_index)
            insert_documents(qasina_index, qasina_docs)
            save_mapping_to_csv(qasina_mapping, 'mappings/qasina_mapping.csv')
        else:
            print("Qasina index already exists")

    except Exception as e:
        print(f"Error while building all index: {e}")
        traceback.print_exc()
