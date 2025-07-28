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
        raise e

def translate_row_musique(data) -> Optional[dict]:
    id = data['id']
    question = data['question']
    contexts = data['context']
    facts = data['supporting_facts']
    answer = data['answer']

    print(f'Translating dataset id: {id}')
    if answer is None:
        return None

    translated_contexts = []
    for title, sentences in zip(contexts['title'], contexts['sentences']):
        translated_title = translate_safe(title)
        translated_sentences = [translate_safe(sentence) for sentence in sentences]
        translated_contexts.append({'title': translated_title, 'sentences': translated_sentences})

    translated_facts = [translate_safe(fact) for fact in facts['title']]
    translated_question = translate_safe(question)
    translated_answer = translate_safe(answer)

    row = {
        'id': id,
        'question': translated_question,
        'contexts': translated_contexts,
        'supporting_facts': translated_facts,
        'answer': translated_answer
    }

    print(f'Done translating dataset {id}')
    
    return row

def translate_multihop_iteration(
        dataset: Dataset,
        testing: bool = False,
        debug_row: Optional[int] = None,
        loaded_dataset: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []

    if debug_row is not None:
        print(f'Debugging row: {debug_row}')

        dataset = dataset.select([debug_row])
        row_debugging = translate_row_musique(dataset[0])

        rows.append(row_debugging)
        return pd.DataFrame(rows)

    ids = loaded_dataset['id'].values if loaded_dataset is not None else []
    
    for index, data in enumerate(dataset):
        id = data['id']
        
        if id in ids:
            print(f"Skipping already translated id: {id}")
            continue
        elif index == 17:
            continue
        elif testing and len(rows) > 2:
            print('Testing done')
            break

        try:
            translated_row = translate_row_musique(data)
            rows.append(translated_row)
        except Exception as e:
            print(f"Error translating row {index} with id {id}: {e}")
            continue

    return pd.DataFrame(rows)

def get_translated_partition(partition: list[str]) -> dict:
    folder_path = 'musique'
    loaded_datasets = {}

    for dataset_name in partition:
        try:
            df = pd.read_csv(f'{folder_path}/{dataset_name}.csv')

            loaded_datasets[dataset_name] = df
        except FileNotFoundError:
            print(f"File not found for dataset: {dataset_name}")
            continue
    
    return loaded_datasets

def save_dataset(
    partition_name: str,
    current_dataset: pd.DataFrame,
    translated_dataset: Optional[pd.DataFrame] = None) -> bool:
    try:
        folder_path = 'musique'
        if translated_dataset is None:
            current_dataset.to_csv(f'{folder_path}/{partition_name}.csv', index=False)
            return True
        else:
            full_dataset = pd.concat([current_dataset, translated_dataset])
            full_dataset.to_csv(f'{folder_path}/{partition_name}.csv', index=False)
            return True
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred while saving the dataset: {e}")
        return False

def translate_multihop(partition: list[str], testing:bool=False, debug_row: Optional[int] = None) -> bool:
    try:
        hotpot_qa = load_dataset('hotpot_qa', 'fullwiki', trust_remote_code=True)
        loaded_datasets = get_translated_partition(partition)
        for dataset_name in partition:
            try:
                dataset = hotpot_qa[dataset_name]
                loaded_dataset = loaded_datasets.get(dataset_name)
                translated_df = translate_multihop_iteration(
                    dataset=dataset, testing=testing, debug_row=debug_row, loaded_dataset=loaded_dataset)

                saving_result = save_dataset(
                    partition_name=dataset_name,
                    current_dataset=translated_df,
                    translated_dataset=loaded_dataset
                )

                if not saving_result:
                    print(f"Failed to save dataset: {dataset_name}")

            except KeyError as e:
                raise e

        print("Translation and saving completed successfully.")
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    print("Running translation script")
    partition = ['validation']
    translate_multihop(
        partition=partition, 
        testing=False, 
        debug_row=None
    )
