from datasets import load_dataset, concatenate_datasets
from vector_database.database_handler import DatabaseHandler
from classification import gather_indo_qa
import pandas as pd
import time

class SeedHandler:
    def __init__(self):
        self.database_handler = DatabaseHandler()
        self.load_dataset()
    
    def load_dataset(self):
        self.wikipedia_dataset = load_dataset("indonesian-nlp/wikipedia-id")
        self.indoqa_training, self.indoqa_validation = gather_indo_qa(False)
    
    def seed(self):
        # self.insert_wikipedia_data()
        self.insert_indoqa_context()
    
    def insert_wikipedia_data(self, n: int = 10000):
        print("Inserting wikipedia data into the vector database...")
        start_time = time.time()
        collection = self.database_handler.get_or_create_collection(
            name="wikipedia_id",
            description="Indonesian Wikipedia dataset"
        )

        full_dataset = concatenate_datasets([
            self.wikipedia_dataset['train'],
            self.wikipedia_dataset['test'],
        ])

        docs = [f"{row['title']} - {row['text']}" for row in full_dataset]
        ids = [f"{row['docid']}" for row in full_dataset]
        metadatas = [{
            "source": "wikipedia_id",
            "title": row['title'],
            "docid": row['docid'],
        } for row in full_dataset]

        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )

        elapsed_time = time.time() - start_time
        print(f"Success inserting wikipedia data in {elapsed_time:.2f} seconds")
    
    def insert_indoqa_context(self):
        print("Inserting indoqa context into the vector database...")
        start_time = time.time()

        # Combine datasets and remove duplicate contexts
        full_df = pd.concat([self.indoqa_training, self.indoqa_validation], ignore_index=True)
        full_df = full_df.drop_duplicates(subset=['context'])

        # Prepare documents, ids, and metadata
        docs = full_df['context'].tolist()
        ids = full_df['id'].tolist()
        metadatas = [{
            "source": "indo_qa_context",
            "title": '',
            "docid": doc_id,
        } for doc_id in ids]

        # Insert into vector database
        collection = self.database_handler.get_or_create_collection(
            name="indo_qa_context",
            description="Indonesian Question Answering Context"
        )

        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )

        elapsed_time = time.time() - start_time
        print(f"Success inserting indoqa context in {elapsed_time:.2f} seconds")
