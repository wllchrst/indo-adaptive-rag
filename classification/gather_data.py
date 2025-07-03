import pandas as pd
from datasets import load_dataset

indo_qa_path = "jakartaresearch/indoqa"

def gather_indo_qa() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = load_dataset(indo_qa_path)

    train_df = pd.DataFrame(ds['train'])
    test_df = pd.DataFrame(ds['validation'])
    
    keep_column = ['id', 'question', 'answer', 'category']

    test_df = test_df[keep_column]
    train_df = train_df[keep_column]

    return train_df, test_df