import pandas as pd
import traceback
import os
from classification.gather_data import gather_indo_qa
from methods import NonRetrieval, SingleRetrieval, MultistepRetrieval
from helpers import EvaluationHelper
from typing import Optional, List

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

def save_classification_result(model_type: str,
                               dataset_name: str,
                               dataset_partition: str,
                               testing: bool,
                               dataset: pd.DataFrame) -> bool:
    try:
        if testing == True:
            print("Is testing classification, not saving any result")
            return True

        folder_save_path = f'{save_path}/{model_type}/{dataset_name}_{dataset_partition}.csv'
        os.makedirs(folder_save_path, exist_ok=True)
        dataset.to_csv(folder_save_path, index=False)

        print(f'Classification result saved at {folder_save_path}')
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

def classify(question: str,
                answer: str,
                logging_classification: bool = False,
                log_method: bool = False,
                index: str = '') -> str:

    non_retrieval_prediction = get_answer(question, non_retrieval, log_method, index)
    single_retrieval_prediction = get_answer(question, single_retrieval, log_method, index)
    
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

    if multistep_retrieval_prediction is not None:
        multi_retrieval_result = EvaluationHelper.compute_scores(answer, multistep_retrieval_prediction)
    
    if logging_classification:
        print(f'Multistep Retrieval: {multistep_retrieval_prediction}')
    
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
