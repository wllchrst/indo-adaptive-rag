from classification import classify_indo_qa

def seed_data():
    from vector_database import SeedHandler
    try:
        handler = SeedHandler()
        handler.database_handler.delete_all_collections()
        handler.seed()
    except Exception as e:
        print(f"Error seeding data: {e}")
    
def test_querying(query: str):
    from vector_database import DatabaseHandler
    
    handler = DatabaseHandler()    
    result = handler.query(collection_name='indo_qa_context', query=query, total_result=3)

    print("*" * 50)
    print(f"Query: {query}")
    for doc, dist in zip(result['documents'][0], result['distances'][0]):
        print(f'- {doc} - {dist}')

    # print(result)

def main():
    # If this is your first time running the application you should 
    # seed all the data into the vector database for context
    seed_data()
    
    # classify_indo_qa()
    # query = 'Dengan siapa Chaerul Saleh, Sukarni, Wikana, dan para pemuda pejuang berdiskusi?'
    # clean_query = 'Chaerul Saleh, Sukarni, Wikana'

    # test_querying(query)
    # test_querying(clean_query)
    
if __name__ == "__main__":
    main()