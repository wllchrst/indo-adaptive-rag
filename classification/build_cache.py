import pandas as pd
from classification.gather_data import gather_indo_qa
from helpers import WordHelper
from bm25 import ElasticsearchRetriever

def build_cache_elastic(index: str):
    train_df, test_df = gather_indo_qa()
    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

    for _, row in full_df.iterrows():
        query = row['question']
        query = WordHelper.clean_sentence(WordHelper.remove_stop_words(query))
        ElasticsearchRetriever.search(
            query=query,
            index=index,
            total_result=3
        )
