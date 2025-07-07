import chromadb
from chromadb.errors import InvalidCollectionException
import datetime
from vector_database.custom_embedder import IndoEmbeddingFunction

class DatabaseHandler:
    def __init__(self):
        self.embedding_function = IndoEmbeddingFunction()
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
        except InvalidCollectionException as e:
            print(f"Error: {e}\nCreating new collection: {name}")
            
            collection = self.client.create_collection(
                    name=name, 
                    metadata={
                        "description": description,
                        "created": str(datetime.datetime.now())
                    },
                    embedding_function=self.embedding_function
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
    
    def get_collections(self) -> list[chromadb.Collection] :
        """Get a list of all collections in the Chroma database."""
        return self.client.list_collections()
    
    def delete_all_collections(self):
        """Delete all collections in the Chroma database."""
        collections = self.get_collections()
        for collection in collections:
            try:
                self.client.delete_collection(name=collection.name)
                print("Deleted collection:", collection.name)
            except InvalidCollectionException:
                print(f"Collection {collection.name} not found.")
            except Exception as e:
                print(f"Error deleting collection {collection.name}: {e}")