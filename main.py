from helpers import env_helper
import argparse
from typing import Optional

parser = argparse.ArgumentParser(description="Python script that is used for indo adaptive rag experiments")


def parse_all_args():
    parser.add_argument("--action", help="Action that is going to be done")
    parser.add_argument("--dataset", help="Dataset name that is going to be classify")
    parser.add_argument("--partition", help="Partition that is going to be run", default='full')
    parser.add_argument("--from", dest="index_from", type=int, default=None,
                        help="Start index for slicing (default: None, meaning start from the beginning)")
    parser.add_argument("--to", dest="index_to", type=int, default=None,
                        help="End index for slicing (default: None, meaning until the end)")
    parser.add_argument("--testing", help="Is the script going to be run for only testing?", action='store_true')
    parser.add_argument("--context", help="Supporting facts to be used for retrieving", action='store_true')
    parser.add_argument("--undersample", help="Undersample the dataset for training", action='store_true')

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


def run_classification_indoqa(partition: str,
                              index_from: Optional[int],
                              index_to: Optional[int]):
    print(f'Running classification for indoqa dataset {partition}')
    from classification import classify_indo_qa
    classify_indo_qa(
        testing=False,
        log_classification=True,
        log_method=False,
        partition=partition,
        index_from=index_from,
        index_to=index_to)


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


def run_classification_qasina(testing: bool = False, context: bool = False):
    print(f'Running classification for qasina dataset')
    print(f'Testing: {testing}')
    from classification import classify_qasina
    classify_qasina(
        testing=testing,
        log_classification=True,
        log_method=False,
    )


def run_translation_script(partition: str, testing: bool):
    print(f'Running translation script for musique dataset {partition}')
    from translation_script import translate_multihop

    translate_multihop(
        partition=[partition],
        testing=testing
    )


def run_train_classifier(undersample: bool):
    from training_classifier import TrainClassifier
    filepath = 'classification_result/final_dataset_augmented.csv'

    train_classifier = TrainClassifier(undersample, filepath)
    model_path = 'indobenchmark/indobert-base-p1'

    train_classifier.train_model(
        training_dataset=train_classifier.train_dataset,
        validation_dataset=train_classifier.val_dataset,
        testing_dataset=train_classifier.test_dataset,
        model_path=model_path
    )


def merge_dataset():
    import os
    from classification import merge_indoqa_dataset

    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "classification_result", "default")

    merge_indoqa_dataset(folder_path,
                         first_file='indoqa__train_101.csv',
                         second_file='indoqa__train_102.csv')


def augment_dataset():
    import asyncio
    from training_classifier import DataLoader
    loader = DataLoader(undersample=False)

    augment_task = loader.augment_dataset(
        dataset=loader.dataset,
        classname='A'
    )

    result = asyncio.run(augment_task)
    print(result['classification'].value_counts())
    result.to_csv('augmented_dataset.csv')


def main():
    arguments = parse_all_args()

    if arguments.action == 'train-classifier':
        run_train_classifier(arguments.undersample)
        return
    elif arguments.action == 'seed_context':
        build_elasticsearch_index()
        return
    elif arguments.action == 'test_context':
        test_querying_elastic(index=arguments.dataset)
        return
    elif arguments.action == 'augment_dataset':
        augment_dataset()
        return

    if arguments.dataset is None or arguments.action is None:
        raise ValueError("Arguments for dataset and action is mandatory")

    elif arguments.action == 'classification':

        if arguments.dataset == 'musique':
            run_classification_musique(arguments.partition, arguments.context, arguments.testing)
        elif arguments.dataset == 'indoqa':
            run_classification_indoqa(arguments.partition, arguments.index_from, arguments.index_to)
        elif arguments.dataset == 'qasina':
            run_classification_qasina(testing=arguments.testing, context=arguments.context)

    elif arguments.action == 'translation':
        run_translation_script(arguments.partition, arguments.testing)
    else:
        raise ValueError(f"Nothing can be run from your arguments {arguments}")


if __name__ == "__main__":
    main()
