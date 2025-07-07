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

        classify(question, answer, False)
        count += 1
        if count == 3:
            return

def classify(question: str, answer: str, logging: bool = False) -> str:
    non_retrieval_prediction = get_answer(question, non_retrieval)
    single_retrieval_prediction = get_answer(question, single_retrieval)

    non_retrieval_result = EvaluationHelper.compute_scores(answer, non_retrieval_prediction)
    single_retrieval_result = EvaluationHelper.compute_scores(answer, single_retrieval_prediction)

    if logging:
        print("*" * 35)
        print(f'Question: {question}')
        print(f'Actual answer: {answer}')
        print(f'No Retrieval: {non_retrieval_prediction}')
        print(f'Single Retrieval: {single_retrieval_prediction}')

    if non_retrieval_result['exact_match'] == 1:
        return 'A'
    elif non_retrieval_result['f1_score'] > single_retrieval_result['f1_score']:
        return 'A'
    else:
        return 'B'
    
def get_answer(question: str, mode: str) -> str:
    if mode not in methods:
        raise ValueError(f"Invalid mode: {mode}. Available modes: {', '.join(methods.keys())}")
    return methods[mode].answer(question, True)