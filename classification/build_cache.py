import pandas as pd
import time
from classification.gather_data import gather_indo_qa
from helpers import WordHelper
from bm25 import ElasticsearchRetriever

## Without cache: 255 seconds
## With cache   : 74 seconds
def build_cache_elastic(index: str):
    start_time = time.time()  # Start the timer

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

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time to build cache: {total_time:.2f} seconds")
