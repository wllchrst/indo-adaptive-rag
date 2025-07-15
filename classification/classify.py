import pandas as pd
import traceback
from classification.gather_data import gather_indo_qa
from methods import NonRetrieval, SingleRetrieval, MultistepRetrieval
from helpers import EvaluationHelper

non_retrieval = 'non-retrieval'
single_retrieval = 'single-retrieval'
multistep_retrieval = 'multistep-retrieval'

# model_type = 'hugging_face'
model_type = 'gemini'
methods = {
    non_retrieval: NonRetrieval(model_type),
    single_retrieval: SingleRetrieval(model_type),
    multistep_retrieval: MultistepRetrieval(model_type)
}

def classify_indo_qa(testing: bool, log_classification: bool):
    try: 
        train_df, test_df = gather_indo_qa()
        full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
        classifications = []

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
        full_df.to_csv('classification_result/indoqa_classified.csv')
        
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
        print("*" * 35)
        print(f'Question: {question}')
        print(f'Actual answer: {answer}')
        print(f'No Retrieval: {non_retrieval_prediction}')
        print(f'Single Retrieval: {single_retrieval_prediction}')

    if non_retrieval_result['exact_match'] == 1:
        return 'A'
    elif single_retrieval_result['exact_match'] == 1:
        return 'B'
    elif non_retrieval_result['f1_score'] > single_retrieval_result['f1_score']:
        return 'A'
    
    multistep_retrieval_prediction = get_answer(question, multistep_retrieval, log_method, index)
    multi_retrieval_result = EvaluationHelper.compute_scores(answer, multistep_retrieval_prediction)
    
    if logging_classification:
        print(f'Multistep Retrieval: {multistep_retrieval_prediction}')
    
    if multi_retrieval_result['exact_match'] == 1:
        return 'C'
    elif single_retrieval_result['f1_score'] > multi_retrieval_result['f1_score']:
        return 'B'
    else:
        return 'C'
    
def get_answer(question: str, mode: str, log_method: bool, index: str) -> str:
    if mode not in methods:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {', '.join(methods.keys())}")
    return methods[mode].answer(question, log_method, index)