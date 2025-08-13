from helpers import env_helper
import argparse
from bm25.build_indexes import build_all_index

parser = argparse.ArgumentParser(description="Python script that is used for indo adaptive rag experiments")


def parse_all_args():
    parser.add_argument("--action", help="Action that is going to be done")
    parser.add_argument("--dataset", help="Dataset name that is going to be classify")
    parser.add_argument("--partition", help="Partition that is going to be run", default='full')
    parser.add_argument("--testing", help="Is the script going to be run for only testing?", action='store_true')
    parser.add_argument("--context", help="Supporting facts to be used for retrieving", action='store_true')

    return parser.parse_args()


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


def test_querying_elastic(index: str = 'indoqa'):
    from bm25 import ElasticsearchRetriever
    retriever = ElasticsearchRetriever()

    print(f"Testing querying elasticsearch index: {index}")
    results = retriever.search_all(index)

    print(results)


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


def run_classification_indoqa(partition: str):
    print(f'Running classification for indoqa dataset {partition}')
    from classification import classify_indo_qa
    classify_indo_qa(
        testing=False,
        log_classification=True,
        log_method=False,
        partition=partition)


def run_classification_musique(partition: str,
                               context: bool,
                               testing: bool = False):
    print(f'Running classification for musique dataset {partition}')
    print(f'Testing: {testing}')
    from classification import classify_musique
    classify_musique(
        testing=testing,
        partition=partition,
        logging_classification=True,
        log_method=False,
        uses_context=context,
    )


def run_translation_script(partition: str, testing: bool):
    print(f'Running translation script for musique dataset {partition}')
    from translation_script import translate_multihop

    translate_multihop(
        partition=[partition],
        testing=testing
    )


def run_train_classifier():
    from training_classifier import TrainClassifier
    train_classifier = TrainClassifier()

    train_classifier.train_model(
        training_dataset=train_classifier.train_dataset,
        validation_dataset=train_classifier.val_dataset,
        testing_dataset=train_classifier.test_dataset,
        model_path='indobenchmark/indobert-base-p1'
    )


def main():
    arguments = parse_all_args()

    if arguments.action == 'train':
        run_train_classifier()
        return
    elif arguments.action == 'seed_context':
        build_elasticsearch_index()
        return
    elif arguments.action == 'test_context':
        test_querying_elastic(index=arguments.dataset)
        return

    if arguments.dataset is None or arguments.action is None:
        raise ValueError("Arguments for dataset and action is mandatory")

    elif arguments.action == 'classification':

        if arguments.dataset == 'musique':
            run_classification_musique(arguments.partition, arguments.context, arguments.testing)
        elif arguments.dataset == 'indoqa':
            run_classification_indoqa(arguments.partition)

    elif arguments.action == 'translation':
        run_translation_script(arguments.partition, arguments.testing)
    else:
        raise ValueError(f"Nothing can be run from your arguments {arguments}")


if __name__ == "__main__":
    main()
