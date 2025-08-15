import pandas as pd
import traceback
import os
import re
import ast
from classification.gather_data import gather_indo_qa, gather_musique_data, gather_qasina_data
from methods import NonRetrieval, SingleRetrieval, MultistepRetrieval
from helpers import EvaluationHelper
from typing import Optional, List
from helpers.word_helper import WordHelper

non_retrieval = 'non-retrieval'
single_retrieval = 'single-retrieval'
multistep_retrieval = 'multistep-retrieval'
save_path = 'classification_result'

# model_type = 'hugging_face'
model_type = 'gemini'
# model_type = 'gemma3:latest'

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


def generate_file_path(model_type: str,
                       dataset_name: str,
                       dataset_partition: str,
                       testing: bool) -> str:
    # Sanitize components
    model_type_sanitized = sanitize_filename(model_type)
    dataset_name_sanitized = sanitize_filename(f'{dataset_name}_{'testing' if testing else ""}')
    dataset_partition_sanitized = sanitize_filename(dataset_partition)
    folder_path = f'{save_path}/{model_type_sanitized}'

    os.makedirs(folder_path, exist_ok=True)
    file_path = f'{folder_path}/{dataset_name_sanitized}_{dataset_partition_sanitized}.csv'

    return file_path


def save_classification_result(model_type: str,
                               dataset_name: str,
                               dataset_partition: str,
                               testing: bool,
                               dataset: pd.DataFrame,
                               old_dataset: Optional[pd.DataFrame]) -> bool:
    try:
        file_path = generate_file_path(model_type, dataset_name, dataset_partition, testing)

        if old_dataset is None:
            dataset.to_csv(file_path, index=False)
            print(f'Saved Total Row: {len(dataset)}')
        else:
            full_dataset = pd.concat([old_dataset, dataset])
            full_dataset.to_csv(file_path, index=False)
            print(f'Saved Total Row: {len(full_dataset)}')

        print(f'Classification result saved at {file_path}')
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error saving classification result: {e}")
        return False


def classify_qasina(testing: bool,
                    log_classification: bool,
                    log_method: bool):
    try:
        dataset_name = 'qasina'
        partition = 'full'
        previous_result: Optional[pd.DataFrame] = None
        df = gather_qasina_data()
        file_path = generate_file_path(model_type, dataset_name, partition, testing)

        if os.path.exists(file_path):
            previous_result = pd.read_csv(file_path)

        classifications = []
        classified_index = []
        ids = [] if previous_result is None else previous_result['ID'].values

        for index, row in df.iterrows():
            try:
                id = row['ID']
                question = row['question']
                answer = row['answer']

                if id in ids:
                    print(f'Skipping ID {id}')
                    continue

                if question is None or answer is None:
                    print(f"Skipping row {index} due to missing question or answer.")
                    continue

                classification_result = classify(
                    question=question,
                    answer=answer,
                    logging_classification=log_classification,
                    log_method=log_method,
                    index='qasina'
                )

                classified_index.append(index)
                classifications.append(classification_result)

                if len(classifications) == 3 and testing:
                    break
            except Exception as e:
                print(f'Classification failed on index {index}: {e}')
                traceback.print_exc()
                break

        df = df.loc[classified_index].copy()
        df['classifications'] = classifications

        save_classification_result(
            model_type=model_type,
            testing=testing,
            dataset_partition=partition,
            dataset=df,
            dataset_name=dataset_name,
            old_dataset=previous_result
        )

        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error classifying qasina: {e}")
        return False


def classify_indo_qa(testing: bool,
                     log_classification: bool,
                     log_method: bool,
                     partition: str = 'full',
                     model_type: str = 'default'):
    try:
        train_df, test_df = gather_indo_qa()
        full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

        if partition == 'train':
            print(f"Training Length: {len(train_df)}")
            full_df = train_df
        elif partition == 'test':
            full_df = test_df

        file_path = generate_file_path(model_type, 'indoqa', partition, testing)
        previous_result: Optional[pd.DataFrame] = None
        if os.path.exists(file_path):
            previous_result = pd.read_csv(file_path)

        ids = [] if previous_result is None else previous_result['id'].values

        classifications = []
        classified_index = []

        for i, row in full_df.iterrows():
            try:
                qid = row['id'] if 'id' in row else i  # fallback if no 'id' column
                question = row['question']
                answer = row['answer']

                if question is None or answer is None:
                    print(f"Skipping row {i} due to missing question or answer.")
                    continue

                if qid in ids:
                    print(f"Skipping ID {qid}")
                    continue

                classification_result = classify(
                    question=question,
                    answer=answer,
                    logging_classification=log_classification,
                    log_method=log_method,
                    index='indoqa'
                )

                classifications.append(classification_result)
                classified_index.append(i)

                # Early stop if testing
                if testing and len(classifications) >= 3:
                    break

            except Exception as e:
                print(f"Classification failed on index {i}: {e}")
                traceback.print_exc()
                break

        # If nothing classified, skip saving
        if not classifications:
            print("Not saving anything — no successful classifications.")
            return True

        # Prepare dataframe for saving
        result_df = full_df.loc[classified_index].copy()
        result_df['classification'] = classifications

        # Save results incrementally
        save_classification_result(
            model_type=model_type,
            testing=testing,
            dataset_partition=partition,
            dataset=result_df,
            dataset_name='indoqa',
            old_dataset=previous_result
        )

        return True

    except Exception as e:
        traceback.print_exc()
        print(f"Error classifying indoqa: {e}")
        return False


def run_classification_on_musique(df,
                                  model_type: str,
                                  dataset_partition: str,
                                  testing: bool,
                                  logging_classification: bool,
                                  log_method: bool,
                                  index: str,
                                  uses_context: bool = False):
    """
    Helper function to classify rows in a dataframe and save the result.
    """
    try:
        previous_result: Optional[pd.DataFrame] = None
        file_path = generate_file_path(model_type, 'musique', dataset_partition, testing)
        if os.path.exists(file_path):
            previous_result = pd.read_csv(file_path)

        ids = [] if previous_result is None else previous_result['id'].values
        classifications = []
        classified_index = []
        for i, row in df.iterrows():
            try:
                id = row['id']
                question = row['question']
                answer = row['answer']
                sf_as_string = row['supporting_facts']
                supporting_facts = ast.literal_eval(sf_as_string)

                if id in ids:
                    print(f'Skipping ID {id}')
                    continue

                classification_result = classify(
                    question=question,
                    answer=answer,
                    logging_classification=logging_classification,
                    log_method=log_method,
                    index=index,
                    supporting_facts=supporting_facts if uses_context else []
                )

                classifications.append(classification_result)
                classified_index.append(i)

                if testing and len(classifications) > 5:
                    break
            except Exception as e:
                print(f'Classification failed on index {i}: {e}')
                traceback.print_exc()
                break

        df = df.loc[classified_index].copy()
        df['classifications'] = classifications

        save_classification_result(
            model_type=model_type,
            testing=testing,
            dataset_partition=dataset_partition,
            dataset=df,
            dataset_name='musique',
            old_dataset=previous_result
        )

        return True
    except Exception as e:
        print(f"Error while classifying musique: {e}")
        traceback.print_exc()
        return False


def classify_musique(testing: bool,
                     partition: str = 'all',
                     uses_context: bool = True,
                     logging_classification: bool = False,
                     log_method: bool = False):
    """
    Main function to classify musique dataset.
    """
    train_df, validation_df = gather_musique_data(partition)
    index = 'musique'

    if validation_df is not None:
        run_classification_on_musique(
            df=validation_df,
            model_type=model_type,
            dataset_partition='validation',
            testing=testing,
            logging_classification=logging_classification,
            log_method=log_method,
            index=index,
            uses_context=uses_context
        )

    if train_df is not None:
        run_classification_on_musique(
            df=train_df,
            model_type=model_type,
            dataset_partition='train',
            testing=testing,
            logging_classification=logging_classification,
            log_method=log_method,
            index=index,
            uses_context=uses_context
        )


def classify(question: str,
             answer: str,
             logging_classification: bool = False,
             log_method: bool = False,
             index: str = '',
             supporting_facts: list[str] = []) -> str:
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

    multistep_retrieval_prediction = get_answer(question, multistep_retrieval, log_method, index, answer,
                                                supporting_facts)

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


def get_answer(question: str, mode: str, log_method: bool, index: str, answer: Optional[str] = None,
               supporting_facts: list[str] = []) -> str:
    if mode not in methods:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {', '.join(methods.keys())}")
    return methods[mode].answer(question, log_method, index, answer, supporting_facts)
