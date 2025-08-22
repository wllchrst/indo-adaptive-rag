import pandas as pd
import traceback
import os
from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentResult:
    dataset_path: str
    dataset_name: str
    file_name: str
    result: pd.DataFrame


def gather_results(result_folder: str = 'experiment_results') -> List[ExperimentResult]:
    dataset_folders = os.listdir(result_folder)
    results: List[ExperimentResult] = []

    for dataset_folder in dataset_folders:
        if os.path.isfile(dataset_folder):
            print(f'Not parsing {dataset_folder} because it is not a folder')
            continue

        folder_path = os.path.join(result_folder, dataset_folder)
        dataset_filenames = os.listdir(folder_path)
        for filename in dataset_filenames:
            if '.csv' not in filename:
                continue

            dataset_path = os.path.join(folder_path, filename)
            df = pd.read_csv(dataset_path)

            experiment_result = ExperimentResult(
                dataset_path=dataset_path,
                dataset_name=dataset_folder,
                file_name=filename,
                result=df
            )

            results.append(experiment_result)

    return results


def get_method_from_filename(filename: str) -> str:
    # remove extension first
    name, _ = os.path.splitext(filename)
    # split by underscore
    parts = name.split("_")
    # method is the last two parts joined with underscore
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return ""


def calculate_all_result(result_folder: str = 'experiment_results') -> bool:
    try:
        conclusions = []
        result_information = gather_results(result_folder)
        for info in result_information:
            data = info.result

            numeric_cols = data.select_dtypes(include=["number"]).columns.difference(["dataset_id"])
            data[numeric_cols] = data[numeric_cols].astype(int)

            exact_match_total = data['exact_match'].sum()
            f1_mean = data['f1_score'].mean()
            step_mean = data['step'].mean()
            time_mean = data['time'].mean()

            llm_model = "Gemma 3" if 'gemma3' in info.file_name else "Gemini"

            conclusion = {
                "method": get_method_from_filename(info.file_name),
                "llm_model": llm_model,
                "dataset": info.dataset_name,
                "exact_match": exact_match_total,
                "f1_mean": f1_mean,
                "step_mean": step_mean,
                "time_mean": time_mean
            }

            conclusions.append(conclusion)

        conclusion_df = pd.DataFrame(conclusions)

        print(conclusion_df.head())
        save_path = os.path.join(result_folder, "final_report.csv")
        conclusion_df.to_csv(save_path)
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error calculating all result {e}")
        return False
