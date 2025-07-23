import pandas as pd
import traceback
from transformers import pipeline
from datasets import load_dataset, Dataset

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id", max_length=2000)

def translate_multihop_iteration(dataset: Dataset, testing: bool = False) -> pd.DataFrame:
    rows = []
    for data in dataset:
        id = data['id']
        question = data['question']
        contexts = data['context']
        facts = data['supporting_facts']
        answer = data['answer']

        print(f'id: {id}')

        translated_contexts = []
        for title, sentences in zip(contexts['title'], contexts['sentences']):
            translated_title = pipe(title)[0]['translation_text']
            translated_sentences = [pipe(sentence)[0]['translation_text'] for sentence in sentences]
            translated_contexts.append({'title': translated_title, 'sentences': translated_sentences})

        translated_facts = [pipe(fact)[0]['translation_text'] for fact in facts['title']]
        translated_question = pipe(question)[0]['translation_text']
        translated_answer = pipe(answer)[0]['translation_text']

        rows.append({
            'id': id,
            'question': translated_question,
            'contexts': translated_contexts,
            'supporting_facts': translated_facts,
            'answer': translated_answer
        })

        if testing and len(rows) > 5:
            break

    return pd.DataFrame(rows)

def translate_multihop(testing:bool=False) -> bool:
    try:
        hotpot_qa = load_dataset('hotpot_qa', 'fullwiki', trust_remote_code=True)
        training, validation, testing = hotpot_qa['train'], hotpot_qa['validation'], hotpot_qa['test']

        training_translated_df = translate_multihop_iteration(training, testing=testing)
        validation_translated_df = translate_multihop_iteration(validation, testing=testing)
        testing_translated_df = translate_multihop_iteration(testing, testing=testing)

        validation_translated_df.to_csv('.musique/validation.csv')
        testing_translated_df.to_csv('.musique/testing.csv')
        training_translated_df.to_csv('./musique/training.csv')

        print("Translation and saving completed successfully.")
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    translate_multihop(True)