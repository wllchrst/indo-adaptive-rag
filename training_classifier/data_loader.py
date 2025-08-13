import pandas as pd
import traceback
import os
from datasets import Dataset
from sklearn.utils import resample


class DataLoader:
    def __init__(self):
        self.dataset = self.gather_all_dataset('classification_result')
        self.dataset = self.undersample_dataset(self.dataset)

    def undersample_dataset(self, hf_dataset: Dataset):
        """
                Perform undersampling so all classes have equal representation.
                Works with Hugging Face Dataset object.
                """
        try:
            # Convert to pandas for resampling
            df = hf_dataset.to_pandas()

            # Find minimum class count
            min_count = df['classification'].value_counts().min()

            # Perform undersampling for each class
            df_balanced = (
                df.groupby('classification', group_keys=False)
                .apply(lambda x: resample(x, replace=False, n_samples=min_count, random_state=42))
                .reset_index(drop=True)
            )

            # Convert back to Hugging Face Dataset
            return Dataset.from_pandas(df_balanced)

        except Exception as e:
            traceback.print_exc()
            return hf_dataset

    def gather_all_dataset(self, folder_dataset: str) -> Dataset:
        try:
            joined_dataset = None
            folders = os.listdir(folder_dataset)
            for folder in folders:
                files = os.listdir(folder)
                for filename in files:
                    if '.csv' not in filename:
                        continue

                    dataset = pd.read_csv(filename)
                    joined_dataset = dataset if joined_dataset \
                                                is None else pd.concat([joined_dataset, dataset])

            if joined_dataset is None:
                raise ValueError("No CSV files found with 'question' and 'classification' columns.")

            joined_dataset.reset_index(drop=True, inplace=True)

            hf_dataset = Dataset.from_pandas(joined_dataset)
            return hf_dataset

        except Exception as e:
            traceback.print_exc()
            raise e
