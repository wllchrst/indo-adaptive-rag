from classification.gather_data import gather_indo_qa
from methods import NonRetrieval, SingleRetrieval, MultistepRetrieval
from helpers import EvaluationHelper

non_retrieval = 'non-retrieval'
single_retrieval = 'single-retrieval'
multistep_retrieval = 'multistep-retrieval'

methods = {
    non_retrieval: NonRetrieval(),
    single_retrieval: SingleRetrieval(),
    multistep_retrieval: MultistepRetrieval()
}

def classify_indo_qa():
    train_df, test_df = gather_indo_qa()
    count = 0

    for index, row in train_df.iterrows():
        question = row['question']
        answer = row['answer']

        classify(
            question=question,
            answer=answer,
            logging_classification=True,
            log_method=False,
            index='indoqa'
        )
        count += 1
        if count == 3:
            return

def classify(question: str,
                answer: str,
                logging_classification: bool = False,
                log_method: bool = False,
                index: str = '') -> str:

    non_retrieval_prediction = get_answer(question, non_retrieval, log_method, index)
    single_retrieval_prediction = get_answer(question, single_retrieval, log_method, index)
    multistep_retrieval_prediction = '' # get_answer(question, multistep_retrieval, log_method, index)

    non_retrieval_result = EvaluationHelper.compute_scores(answer, non_retrieval_prediction)
    single_retrieval_result = EvaluationHelper.compute_scores(answer, single_retrieval_prediction)
    multi_retrieval_result = EvaluationHelper.compute_scores(answer, multistep_retrieval_prediction)

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
    elif multi_retrieval_result['exact_match'] == 1:
        return 'C'
    elif non_retrieval_result['f1_score'] > single_retrieval_result['f1_score']:
        return 'A'
    elif single_retrieval_result['f1_score'] > multi_retrieval_result['f1_score']:
        return 'B'
    else:
        return 'C'
    
def get_answer(question: str, mode: str, log_method: bool, index: str) -> str:
    if mode not in methods:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {', '.join(methods.keys())}")
    return methods[mode].answer(question, log_method, index)