from datasets import load_dataset, concatenate_datasets
from vector_database.database_handler import DatabaseHandler

class SeedHandler:
    def __init__(self):
        self.database_handler = DatabaseHandler()
        self.load_dataset()
        self.seed()
    
    def load_dataset(self):
        self.wikipedia_dataset = load_dataset("indonesian-nlp/wikipedia-id")
    
    def seed(self):
        self.insert_wikipedia_data(n=1000)
    
    def insert_wikipedia_data(self, n: int):
        print("Inserting wikipedia data into the vector database...")
        collection = self.database_handler.get_or_create_collection(
            name="wikipedia_id",
            description="Indonesian Wikipedia dataset"
        )

        full_dataset = concatenate_datasets([
            self.wikipedia_dataset['train'],
            self.wikipedia_dataset['test'],
        ])

        full_dataset = full_dataset.select(range(n))

        docs = [f'{row['title']} - {row['text']}' for row in full_dataset]
        ids = [f'{row['docid']}' for row in full_dataset]
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