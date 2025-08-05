import pandas as pd
import os
from datasets import load_dataset

indo_qa_path = "jakartaresearch/indoqa"
musique_folder_path = "musique"
validation_filename = "validation.csv"
train_filename = "train.csv"

def gather_indo_qa(remove_column: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = load_dataset(indo_qa_path)

    train_df = pd.DataFrame(ds['train'])
    test_df = pd.DataFrame(ds['validation'])
    
    keep_column = ['id', 'question', 'answer', 'category']

    if remove_column:
        test_df = test_df[keep_column]
        train_df = train_df[keep_column]

    return train_df, test_df

def gather_musique_data(partition: str = 'all') -> tuple[pd.DataFrame, pd.DataFrame]:
    folder_exists = os.path.exists(musique_folder_path)
    if not folder_exists:
        raise FileNotFoundError(f"{musique_folder_path} doesn't exists in the current directory")

    filenames = os.listdir(musique_folder_path)
    csv_file_exists = any("csv" in filename for filename in filenames)

    if not csv_file_exists:
        raise FileNotFoundError(f"There is no csv file in {musique_folder_path}")

    validation_filepath = os.path.join(musique_folder_path, validation_filename)
    train_filepath = os.path.join(musique_folder_path, train_filename)

    if partition == 'all':
        train_df = pd.read_csv(train_filepath)
        validation_df = pd.read_csv(validation_filepath)
        return train_df, validation_df
    elif partition == 'train':
        train_df = pd.DataFrame(train_filepath)
        return train_df, None
    else:
        validation_df = pd.read_csv(validation_filepath)
        return None, validation_df