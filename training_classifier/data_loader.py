import pandas as pd
import traceback
import os
from typing import Optional
from datasets import Dataset
from sklearn.utils import resample
from training_classifier.eda import indonesia_eda


class DataLoader:
    def __init__(self,
                 main_folder_dataset: str = 'classification_result/final_dataset',
                 musique_folder_dataset: str = 'musique',
                 undersample: bool = True,
                 file_path: Optional[str] = None):
        self.main_folder_dataset = main_folder_dataset
        self.musique_folder_dataset = musique_folder_dataset
        self.dataset = self.gather_all_dataset(file_path=file_path)

        if undersample:
            self.dataset = self.undersample_dataset(self.dataset)

    async def augment_dataset(self, dataset: Dataset, classname: str) -> pd.DataFrame:
        hf_dataset = dataset.to_pandas()
        print(hf_dataset['classification'].value_counts())

        df_to_augment = hf_dataset[hf_dataset['classification'] == classname]

        augmented_sentences = []

        for index, row in df_to_augment.iterrows():
            sentence = row['question']

            new_sentences = await indonesia_eda(sentence)

            for new_sentence in new_sentences:
                augmented_sentences.append({
                    "question": new_sentence,
                    "classification": row['classification']
                })

        augmented_df = pd.DataFrame(augmented_sentences)

        combined_df = pd.concat([hf_dataset, augmented_df]).reset_index(drop=True)

        return combined_df

    def undersample_dataset(self, hf_dataset: Dataset) -> Dataset:
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

            print(f'Info: {df_balanced.info()}')
            print(df_balanced['classification'].value_counts())

            return Dataset.from_pandas(df_balanced)

        except Exception as e:
            traceback.print_exc()
            return hf_dataset

    def gather_all_dataset(self, file_path: Optional[str] = None) -> Dataset:
        if file_path is not None:
            df = pd.read_csv(file_path)
            return Dataset.from_pandas(df)

        allowed_classes = ["A", "B", "C"]
        main_dataset = self.gather_dataset_from_folder(self.main_folder_dataset)
        musique_dataset = self.gather_musique_dataset(self.musique_folder_dataset)

        dataset = pd.concat([main_dataset, musique_dataset])
        dataset = dataset[dataset["classification"].isin(allowed_classes)]
        dataset = dataset.reset_index(drop=True)

        dataset = Dataset.from_pandas(dataset)

        return dataset

    def gather_dataset_from_folder(self, folder_dataset: str) -> pd.DataFrame:
        try:
            joined_dataset = None
            files = os.listdir(folder_dataset)

            for filename in files:
                if not filename.endswith(".csv"):
                    continue

                filepath = os.path.join(folder_dataset, filename)
                print(f'filepath: {filepath}')
                df = pd.read_csv(filepath, usecols=["question", "classification"])

                joined_dataset = df if joined_dataset is None else pd.concat([joined_dataset, df])

            if joined_dataset is None:
                raise ValueError("No CSV files found with 'question' and 'classification' columns.")

            joined_dataset.reset_index(drop=True, inplace=True)
            return joined_dataset

        except Exception as e:
            traceback.print_exc()
            raise e

    def gather_musique_dataset(self, folder_dataset: str) -> pd.DataFrame:
        try:
            joined_dataset = None
            files = os.listdir(folder_dataset)

            for filename in files:
                if not filename.endswith(".csv"):
                    continue

                filepath = os.path.join(folder_dataset, filename)
                df = pd.read_csv(filepath, usecols=["question"])

                # add classification column with value "C"
                df["classification"] = "C"

                joined_dataset = df if joined_dataset is None else pd.concat([joined_dataset, df])

            if joined_dataset is None:
                raise ValueError("No CSV files found with 'question' column.")

            joined_dataset.reset_index(drop=True, inplace=True)
            return joined_dataset

        except Exception as e:
            traceback.print_exc()
            raise e
