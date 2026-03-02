import pandas as pd
import traceback
import torch
import re
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from typing import Optional
from translator import translate_safe

load_dotenv()


def translate_row_hotpot(data, by_token: bool = False) -> Optional[dict]:
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
        translated_title = translate_safe(title, by_token=by_token)
        translated_sentences = [translate_safe(sentence, by_token=by_token) for sentence in sentences]
        translated_contexts.append({'title': translated_title, 'sentences': translated_sentences})

    translated_facts = [translate_safe(fact, by_token=by_token) for fact in facts['title']]
    translated_question = translate_safe(question, by_token=by_token)
    translated_answer = translate_safe(answer, by_token=by_token)

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
        row_debugging = translate_row_hotpot(dataset[0])

        rows.append(row_debugging)
        return pd.DataFrame(rows)

    ids = loaded_dataset['id'].values if loaded_dataset is not None else []

    skip_indices = [3797, 3850, 3911, 3970, 4025, 4048, 4130, 4526]

    for index, data in enumerate(dataset):
        id = data['id']

        if id in ids:
            print(f"Skipping already translated id: {id}")
            continue
        elif index in skip_indices:
            continue
        elif testing and len(rows) > 2:
            print('Testing done')
            break

        try:
            translated_row = translate_row_hotpot(data)
            rows.append(translated_row)
        except Exception as e:
            print(f"Error translating row {index} with id {id}: {e}")
            return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def get_translated_partition(partition: list[str]) -> dict:
    folder_path = 'hotpot'
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
        folder_path = 'hotpot'
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


def translate_multihop(partition: list[str], testing: bool = False, debug_row: Optional[int] = None) -> bool:
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
                else:
                    print(f'Saved data with name: {dataset_name}')

            except KeyError as e:
                raise e

        print("Translation and saving completed successfully.")
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
        return False


def generate_parallel_validation_sets():
    import json
    
    print("Loading Indonesian validation data...")
    indonesian_df = pd.read_csv('hotpot/validation.csv')
    print(f"Loaded {len(indonesian_df)} rows from Indonesian validation data")
    
    print("Loading English validation data from hotpot_qa dataset...")
    hotpot_qa = load_dataset('hotpot_qa', 'fullwiki', trust_remote_code=True)
    english_data = hotpot_qa['validation']
    print(f"Loaded {len(english_data)} rows from English validation data")
    
    print("Creating English DataFrame...")
    english_list = []
    for row in english_data:
        english_list.append({
            'id': row['id'],
            'question': row['question'],
            'contexts': row['context'],
            'supporting_facts': row['supporting_facts'],
            'answer': row['answer']
        })
    english_df = pd.DataFrame(english_list)
    
    print("Merging English and Indonesian data by ID...")
    merged_df = english_df.merge(indonesian_df, on='id', suffixes=('_en', '_id'))
    print(f"Merged to {len(merged_df)} rows")
    
    print("Creating parallel columns...")
    output_df = pd.DataFrame({
        'id': merged_df['id'],
        'question_en': merged_df['question_en'],
        'question_id': merged_df['question_id'],
        'answer_en': merged_df['answer_en'],
        'answer_id': merged_df['answer_id']
    })
    
    print("Shuffling data...")
    output_df = output_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Splitting into 3 parts (150 rows each)...")
    part1 = output_df.iloc[:150]
    part2 = output_df.iloc[150:300]
    part3 = output_df.iloc[300:450]
    
    print("Saving CSV files...")
    part1.to_csv('hotpot/validation_part1.csv', index=False)
    part2.to_csv('hotpot/validation_part2.csv', index=False)
    part3.to_csv('hotpot/validation_part3.csv', index=False)
    
    print(f"Done! Generated 3 CSV files:")
    print(f"  - hotpot/validation_part1.csv ({len(part1)} rows)")
    print(f"  - hotpot/validation_part2.csv ({len(part2)} rows)")
    print(f"  - hotpot/validation_part3.csv ({len(part3)} rows)")


if __name__ == "__main__":
    print("Running translation script")
    partition = ['train']
    # translate_multihop(
    #     partition=partition,
    #     testing=False,
    #     debug_row=None
    # )

    generate_parallel_validation_sets()
