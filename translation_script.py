import pandas as pd
import traceback
import torch
import re
from transformers import pipeline
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from typing import Optional

load_dotenv()
device_number = 0 if torch.cuda.is_available() else -1
MAX_TOKEN = 300

print(f"Using device: {device_number}")
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id", max_length=2000, device=device_number)

def split_long_text(text, max_length=MAX_TOKEN):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < max_length:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())
    return chunks

def translate_safe(text: str) -> str:
    try:
        if len(text) <= MAX_TOKEN:
            return pipe(text)[0]['translation_text']
        else:
            chunks = split_long_text(text)
            return " ".join(pipe(chunk)[0]['translation_text'] for chunk in chunks)
    except Exception as e:
        print(f"[Translation skipped] Error: {e} | Text: {text[:50]}...")
        return text  # return the original text in case of failure


def translate_row_musique(data) -> Optional[dict]:
    id = data['id']
    question = data['question']
    contexts = data['context']
    facts = data['supporting_facts']
    answer = data['answer']

    if answer is None:
        return None

    translated_contexts = []
    for title, sentences in zip(contexts['title'], contexts['sentences']):
        print(f'Sentences: {sentences}')
        translated_title = pipe(title)[0]['translation_text']
        translated_sentences = [translate_safe(sentence) for sentence in sentences]
        translated_contexts.append({'title': translated_title, 'sentences': translated_sentences})

    translated_facts = [translate_safe(fact) for fact in facts['title']]
    translated_question = pipe(question)[0]['translation_text']
    translated_answer = pipe(answer)[0]['translation_text']

    row = {
        'id': id,
        'question': translated_question,
        'contexts': translated_contexts,
        'supporting_facts': translated_facts,
        'answer': translated_answer
    }
    
    return row

def translate_multihop_iteration(dataset: Dataset, testing: bool = False, debug_row: Optional[int] = None) -> pd.DataFrame:
    rows = []

    if debug_row is not None:
        print(f'Debugging row: {debug_row}')

        dataset = dataset.select([debug_row])
        row_debugging = translate_row_musique(dataset[0])

        rows.append(row_debugging)
        return pd.DataFrame(rows)
    
    for data in dataset:
        translated_row = translate_row_musique(data)

        if testing and len(rows) > 2:
            print('Testing done')
            break

        rows.append(translated_row)

    return pd.DataFrame(rows)

def translate_multihop(partition: list[str], testing:bool=False, debug_row: Optional[int] = None) -> bool:
    try:
        hotpot_qa = load_dataset('hotpot_qa', 'fullwiki', trust_remote_code=True)
        index = 0

        for dataset_name in partition:
            try:
                dataset = hotpot_qa[dataset_name]
                translated_df = translate_multihop_iteration(
                    dataset=dataset, testing=testing, debug_row=debug_row)

                translated_df.to_csv(f'musique/{dataset_name}.csv')
                print(translated_df.head())
            except KeyError as e:
                print(f'Error at row {index}: {e}')
                raise e

        print("Translation and saving completed successfully.")
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    print("Running translation script")
    partition = ['validation', 'test']
    translate_multihop(
        partition=partition, 
        testing=False, 
        debug_row=17
    )