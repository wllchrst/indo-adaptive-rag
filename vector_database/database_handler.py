import chromadb
from chromadb.errors import NotFoundError
import datetime

class DatabaseHandler:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
    
    def get_or_create_collection(self,
                                name: str,
                                description: str = "Default collection description") -> chromadb.Collection:
        """
        Get or create a collection in the Chroma database.
        """
        try:
            find_collection = self.client.get_collection(name=name)
            return find_collection
        except ValueError as e:
            print(f"Error: {e}")
            collection = self.client.create_collection(
                    name=name, 
                    metadata={
                        "description": description,
                        "created": str(datetime.datetime.now())
                    }
                )

            return collection
    
    def check_connection(self) -> int:
        """Check the connection to the Chroma database by sending a heartbeat."""
        return self.client.heartbeat()
    
    def query(self,
            collection_name: str,
            query: str,
            total_result: int) -> chromadb.QueryResult:
        """Query the Chroma database for a specific collection.
        Args:
            collection_name (str): The name of the collection to query.
            query (str): The query string to search for.
            total_result (int): The number of results to return.
        Returns:
            list: A list of documents matching the query.
        """
        
        collection = self.get_or_create_collection(name=collection_name)
        result = collection.query(
            query_texts=[query],
            n_results=total_result,
        )

        return result