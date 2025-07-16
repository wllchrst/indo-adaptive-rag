from helpers import env_helper
from classification.build_cache import build_cache_elastic

def seed_data():
    from vector_database import SeedHandler
    try:
        handler = SeedHandler()
        handler.database_handler.delete_all_collections()
        handler.seed()
    except Exception as e:
        print(f"Error seeding data: {e}")

def build_elasticsearch_index():
    from bm25 import build_all_index
    build_all_index()
    
def test_querying_chroma(query: str):
    from vector_database import DatabaseHandler
    
    handler = DatabaseHandler()    
    result = handler.query(collection_name='indo_qa_context', query=query, total_result=3)

    print("*" * 50)
    print(f"Query: {query}")
    for doc, dist in zip(result['documents'][0], result['distances'][0]):
        print(f'- {doc} - {dist}')

    # print(result)

def test_querying_elastic(query: str):
    from bm25 import ElasticsearchRetriever
    retriever = ElasticsearchRetriever()

    results = retriever.search(
        index='indoqa',
        query=query,
        total_result=5
    )

    for result in results:
        print(result)
        
def test_llm():
    question = "Siapa nama presiden pertama Indonesia"
    try:
        from llm import HuggingFaceLLM
        model_name = 'kalisai/Nusantara-0.8b-Indo-Chat'
        llm = HuggingFaceLLM(model_name=model_name)
        answer = llm.answer(question)
        print(f"Answering Finished: {answer}")
    except Exception as e:
        print(e)

def clear_cache() -> bool:
    try:
        from joblib import Memory
        memory = Memory(location=env_helper.CACHE_DIRECTORY, verbose=0)
        memory.clear()
        return True
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False

def run_classification_indoqa():
    from classification import classify_indo_qa, build_cache_elastic
    # cache_cleared = clear_cache()

    # if not cache_cleared:
    #     print("Classification failed, clearing cache failed")
    #     return

    # build_cache_elastic("indoqa")
    classify_indo_qa(True, True)

def main():
    run_classification_indoqa()
    # test_querying_elastic("Chaerul Saleh")

if __name__ == "__main__":
    print("Script main.py is being run")
    main()