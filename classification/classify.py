import pandas as pd
import traceback
import os
import re
from classification.gather_data import gather_indo_qa, gather_musique_data
from methods import NonRetrieval, SingleRetrieval, MultistepRetrieval
from helpers import EvaluationHelper
from typing import Optional, List
from helpers.word_helper import WordHelper

non_retrieval = 'non-retrieval'
single_retrieval = 'single-retrieval'
multistep_retrieval = 'multistep-retrieval'
save_path = 'classification_result'

#model_type = 'hugging_face'
#model_type = 'gemini'
model_type = 'gemma3:latest'

methods = {
    non_retrieval: NonRetrieval(model_type),
    single_retrieval: SingleRetrieval(model_type),
    multistep_retrieval: MultistepRetrieval(model_type)
}

def sanitize_filename(name: str) -> str:
    """
    Replace characters that are invalid in filenames with underscores.
    This ensures compatibility across all major operating systems.
    """
    # This regex matches characters: \ / : * ? " < > | and control characters (0–31)
    return re.sub(r'[\\/*?:"<>|\x00-\x1F]', '_', name)


def save_classification_result(model_type: str,
                               dataset_name: str,
                               dataset_partition: str,
                               testing: bool,
                               dataset: pd.DataFrame) -> bool:
    try:
        # Sanitize components
        model_type_sanitized = sanitize_filename(model_type)
        dataset_name_sanitized = sanitize_filename(f'{dataset_name}_{testing}')
        dataset_partition_sanitized = sanitize_filename(dataset_partition)

        folder_path = f'{save_path}/{model_type_sanitized}'
        file_path = f'{folder_path}/{dataset_name_sanitized}_{dataset_partition_sanitized}.csv'

        os.makedirs(folder_path, exist_ok=True)
        dataset.to_csv(file_path, index=False)

        print(f'Classification result saved at {file_path}')
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error saving classification result: {e}")
        return False

def classify_indo_qa(testing: bool,
                    log_classification: bool,
                    partition: str = 'full'):
    try: 
        train_df, test_df = gather_indo_qa()
        full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        classifications = []

        if partition == 'train':
            full_df = train_df
        elif partition == 'test':
            full_df = test_df

        for index, row in full_df.iterrows():
            question = row['question']
            answer = row['answer']

            if question is None or answer is None:
                print(f"Skipping row {index} due to missing question or answer.")
                continue

            classification_result = classify(
                question=question,
                answer=answer,
                logging_classification=log_classification,
                log_method=False,
                index='indoqa'
            )

            classifications.append(classification_result)

            if len(classifications) == 3 and testing:
                break

        full_df.loc[:len(classifications)-1, 'classification'] = classifications

        save_classification_result(
            dataset=full_df,
            dataset_name='indoqa',
            testing=testing,
            dataset_partition=partition,
            model_type=model_type
        )
        
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error classifying indoqa: {e}")
        return False

def run_classification_on_musique(df, model_type, dataset_partition, testing, logging_classification, log_method, index):
    """
    Helper function to classify rows in a dataframe and save the result.
    """
    try:
        classifications = []

        for _, row in df.iterrows():
            question = row['question']
            answer = row['answer']

            classification_result = classify(
                question=question,
                answer=answer,
                logging_classification=logging_classification,
                log_method=log_method,
                index=index
            )
            classifications.append(classification_result)

            if testing and len(classifications) > 5:
                break

        df.loc[:len(classifications) - 1, 'classification'] = classifications

        save_classification_result(
            model_type=model_type,
            testing=testing,
            dataset_partition=dataset_partition,
            dataset=df,
            dataset_name='musique'
        )

        return True
    except Exception as e:
        print(f"Error while classifying musique: {e}")
        return False

def classify_musique(testing: bool,
                     partition: str = 'all',
                     uses_context: bool = True,
                     logging_classification: bool = False,
                     log_method: bool = False,
                     index: str = '',
                     model_type: str = 'default'):
    """
    Main function to classify musique dataset.
    """
    train_df, validation_df = gather_musique_data(partition)

    if validation_df is not None:
        print(validation_df.info())
        run_classification_on_musique(
            df=validation_df,
            model_type=model_type,
            dataset_partition='validation',
            testing=testing,
            logging_classification=logging_classification,
            log_method=log_method,
            index=index
        )

    if train_df is not None:
        run_classification_on_musique(
            df=train_df,
            model_type=model_type,
            dataset_partition='train',
            testing=testing,
            logging_classification=logging_classification,
            log_method=log_method,
            index=index
        )

def classify(question: str,
                answer: str,
                logging_classification: bool = False,
                log_method: bool = False,
                index: str = '') -> str:
    non_retrieval_prediction = get_answer(question, non_retrieval, log_method, index)
    if WordHelper.contains(answer, non_retrieval_prediction):
        return 'A'

    single_retrieval_prediction = get_answer(question, single_retrieval, log_method, index)
    if WordHelper.contains(answer, single_retrieval_prediction):
        return 'B'

    non_retrieval_result = EvaluationHelper.compute_scores(answer, non_retrieval_prediction)
    single_retrieval_result = EvaluationHelper.compute_scores(answer, single_retrieval_prediction)
    
    if logging_classification:
        print("*" * 100)
        print(f'Question: {question}')
        print(f'Actual answer: {answer}')
        print(f'No Retrieval {non_retrieval_result}: {non_retrieval_prediction}')
        print(f'Single Retrieval {single_retrieval_result}: {single_retrieval_prediction}')

    if non_retrieval_result['exact_match'] == 1:
        return 'A'
    elif single_retrieval_result['exact_match'] == 1:
        return 'B'
    elif non_retrieval_result['f1_score'] > single_retrieval_result['f1_score']:
        return 'A'
    
    multistep_retrieval_prediction = get_answer(question, multistep_retrieval, log_method, index, answer)

    # If multistep method cannot answer the question, return 'C'
    if multistep_retrieval_prediction is None:
        return 'C'

    multi_retrieval_result = EvaluationHelper.compute_scores(answer, multistep_retrieval_prediction)
    if logging_classification:
        print(f'Multistep Retrieval {multi_retrieval_result}: {multistep_retrieval_prediction}')
    
    if multi_retrieval_result['exact_match'] == 1:
        return 'C'
    elif single_retrieval_result['f1_score'] > multi_retrieval_result['f1_score']:
        return 'B'
    else:
        return 'C'
    
def get_answer(question: str, mode: str, log_method: bool, index: str, answer: Optional[str] = None) -> str:
    if mode not in methods:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {', '.join(methods.keys())}")
    return methods[mode].answer(question, log_method, index, answer)
