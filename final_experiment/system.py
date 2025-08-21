import pandas as pd
import time
import os
import traceback
from final_experiment.classifier import Classifier
from typing import List, Tuple, Dict
from methods import NonRetrieval, SingleRetrieval, MultistepRetrieval
from enum import Enum
from helpers import EvaluationHelper


class SystemType(Enum):
    MULTI = 'MULTISTEP'
    SINGLE = 'SINGLE STEP'
    NON = 'NO STEP'
    ADAPTIVE = 'ADAPTIVE'


system_type_mapping = {
    'non-retrieval': SystemType.NON,
    'single-retrieval': SystemType.NON,
    'multi-retrieval': SystemType.NON,
    'adaptive': SystemType.ADAPTIVE,
}


class System:
    def __init__(self,
                 classifier_model_path: str,
                 dataset_path: str,
                 dataset_index: str,
                 dataset_name: str,
                 dataset_part: float,
                 keep_column: List[str],
                 model_type: str,
                 question_column: str = 'question',
                 answer_column: str = 'answer',
                 id_column: str = 'id',
                 experiment_result_folder: str = 'experiment_results'):
        print("\n🚀 System initialized with configuration:")
        print(f"  classifier_model_path : {classifier_model_path}")
        print(f"  dataset_path          : {dataset_path}")
        print(f"  dataset_index         : {dataset_index}")
        print(f"  dataset_name          : {dataset_name}")
        print(f"  dataset_part          : {dataset_part}")
        print(f"  keep_column           : {keep_column}")
        print(f"  model_type            : {model_type}")
        print(f"  question_column       : {question_column}")
        print(f"  answer_column         : {answer_column}")
        print(f"  id_column             : {id_column}")
        print(f"  experiment_result_dir : {experiment_result_folder}\n")

        self.classifier = Classifier(model_path=classifier_model_path)
        self.type_mapping = {
            "A": SystemType.NON,
            "B": SystemType.SINGLE,
            "C": SystemType.MULTI,
        }

        self.dataset = self.gather_dataset(dataset_path, keep_column, dataset_part)
        self.dataset_index = dataset_index
        self.dataset_name = dataset_name
        self.question_column = question_column
        self.answer_column = answer_column
        self.id_column = id_column

        self.model_type = model_type
        self.non_retrieval = NonRetrieval(model_type)
        self.single_retrieval = SingleRetrieval(model_type)
        self.multi_retrieval = MultistepRetrieval(model_type)
        self.method_map = {
            SystemType.NON: self.non_retrieval,
            SystemType.SINGLE: self.single_retrieval,
            SystemType.MULTI: self.multi_retrieval
        }

        self.experiment_result_folder = experiment_result_folder

    def gather_dataset(self,
                       dataset_path: str,
                       keep_column: List[str],
                       dataset_part: float = 1) -> pd.DataFrame:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File doesn't exists: {dataset_path}")
        elif "csv" not in dataset_path:
            raise FileNotFoundError(f"{dataset_path} is not a .csv file")

        df = pd.read_csv(dataset_path)
        df = df[keep_column]

        if 0 < dataset_part < 1:
            df = df.sample(frac=dataset_part, random_state=42).reset_index(drop=True)
        elif dataset_part == 1:
            df = df.reset_index(drop=True)
        else:
            raise ValueError("dataset_part must be between 0 and 1")

        return df

    def process(self, system_type: SystemType):
        try:
            experiment_result: List[Dict] = []
            file_save_path = self.generate_file_name(system_type)
            existing_result = None
            ids = []

            if os.path.exists(file_save_path):
                existing_result = pd.read_csv(file_save_path)
                ids = existing_result[self.id_column].values

            for index, row in self.dataset.iterrows():
                try:
                    dataset_id = row[self.id_column]
                    if dataset_id in ids:
                        print(f'Skipping row with dataset id: {dataset_id}')
                        continue

                    start_time = time.time()
                    answer, retrieve_count = self.answer_question(
                        question=row[self.question_column],
                        system_type=system_type
                    )

                    end_time = time.time()
                    elapsed = end_time - start_time

                    result = EvaluationHelper.compute_scores(
                        a_gold=row[self.answer_column],
                        a_pred=answer
                    )

                    result['time'] = elapsed
                    result['step'] = retrieve_count
                    result['dataset_id'] = dataset_id
                    experiment_result.append(result)

                    # 🖨️ Print nicely
                    print("\n===============================")
                    print(f"ID       : {dataset_id}")
                    print(f"Question : {row[self.question_column]}")
                    print(f"Gold Ans : {row[self.answer_column]}")
                    print(f"Pred Ans : {answer}")
                    print(f"Scores   : {result}")
                    print("===============================\n")
                except Exception as e:
                    traceback.print_exc()
                    print(f'Error when trying to answer index: {index}')
                    break

            experiment_result_df = pd.DataFrame(experiment_result)
            experiment_result_df.to_csv(file_save_path, index=False)

            if existing_result is not None:
                combined_dataset = pd.concat([existing_result, experiment_result_df], ignore_index=True)
                combined_dataset.drop_duplicates(subset=[self.id_column], inplace=True)
                combined_dataset.to_csv(file_save_path, index=False)

            print(f"Final experiment done: {file_save_path}")
        except Exception as e:
            traceback.print_exc()
            print(f'Error doing final experiment: {e}')

    def answer_question(self,
                        question: str,
                        system_type: SystemType) -> Tuple[str, int]:
        if system_type is SystemType.ADAPTIVE:
            classification = self.classifier.classify(text=question)
            system_type = self.type_mapping.get(classification)

        if system_type is None:
            raise ValueError(f"Unsupported system type: {system_type}")

        retriever = self.method_map.get(system_type)

        if retriever is None:
            raise ValueError(f"Unsupported system type: {system_type}")

        return retriever.answer(
            query=question,
            with_logging=False,
            index=self.dataset_index,
            answer=None,
            supporting_facts=[]
        )

    def generate_file_name(self, system_type: SystemType) -> str:
        folder = f'{self.experiment_result_folder}/{self.dataset_name}'
        os.makedirs(folder, exist_ok=True)

        file_save_path = f'{folder}/{self.model_type}_{system_type}.csv'
        return file_save_path
