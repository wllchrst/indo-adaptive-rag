import pandas as pd
import csv
import time
import os
import traceback
import re
import random
import numpy as np
from pprint import pprint
from scipy.stats import bootstrap as scipy_bootstrap
from typing import Optional, List, Dict
from pathlib import Path
import json

from final_experiment.classifier import Classifier
from typing import List, Tuple, Dict
from methods import NonRetrieval, SingleRetrieval, MultistepRetrieval
from enum import Enum
from helpers import EvaluationHelper, WordHelper


class SystemType(Enum):
    MULTI = 'MULTISTEP'
    SINGLE = 'SINGLE STEP'
    NON = 'NO STEP'
    ADAPTIVE = 'ADAPTIVE'


system_type_mapping = {
    'non-retrieval': SystemType.NON,
    'single-retrieval': SystemType.SINGLE,
    'multi-retrieval': SystemType.MULTI,
    'adaptive': SystemType.ADAPTIVE,
}

retrieval_type_mapping = {
    'single-retrieval': SystemType.SINGLE,
    'multi-retrieval': SystemType.MULTI,
    'adaptive': SystemType.ADAPTIVE,
}

reverse_mapping = {
    SystemType.NON: 'non-retrieval',
    SystemType.SINGLE: 'single-retrieval',
    SystemType.MULTI: 'multi-retrieval',
    SystemType.ADAPTIVE: 'adaptive',
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
                 experiment_result_folder: str = 'experiment_results',
                 n_bootstrap_samples: int = 10,
                 random_seed: int = 42,
                 retrieval_type: str = 'lexical',
                 skip_init: bool = False):
        self.experiment_result_folder = experiment_result_folder

        if skip_init:
            return

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
        print(f"  experiment_result_dir : {experiment_result_folder}")
        print(f"  retrieval_type        : {retrieval_type}\n")

        self.classifier = Classifier(model_path=classifier_model_path)
        self.type_mapping = {
            "A": SystemType.NON,
            "B": SystemType.SINGLE,
            "C": SystemType.MULTI,
        }

        self.dataset_index = dataset_index
        self.dataset_name = dataset_name
        self.question_column = question_column
        self.answer_column = answer_column
        self.id_column = id_column
        self.dataset = self.gather_dataset(dataset_path, keep_column, dataset_part)

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

        self.retrieval_type = retrieval_type

        # Add bootstrap configuration
        self.n_bootstrap_samples = n_bootstrap_samples
        self.random_seed = random_seed

        # Set random seeds for reproducibility
        self._set_random_seeds()

        # Log initial configuration
        self._log_experiment_config()

    def gather_dataset(self,
                       dataset_path: str,
                       keep_column: List[str],
                       dataset_part: float = 1) -> pd.DataFrame:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"File doesn't exists: {dataset_path}")
        elif "csv" not in dataset_path:
            raise FileNotFoundError(f"{dataset_path} is not a .csv file")

        try:
            df = pd.read_csv(dataset_path)
            if self.dataset_name == 'qasina':
                df[self.id_column] = pd.to_numeric(df[self.id_column], errors="coerce")
                df = df.dropna(subset=[self.id_column])
                df[self.id_column] = df[self.id_column].astype(int)

        except Exception as e:
            print(f'Error when trying to fix id_column: {self.id_column}')
            traceback.print_exc()
            raise e

        df = df[keep_column]

        if 0 < dataset_part < 1:
            df = df.sample(frac=dataset_part, random_state=42).reset_index(drop=True)
        elif dataset_part == 1:
            df = df.reset_index(drop=True)
        else:
            raise ValueError("dataset_part must be between 0 and 1")

        print(f'✅ Dataset total row: {len(df)}')

        return df

    def process(self, system_type: SystemType):
        try:
            print(f"🥸 Running process using type: {system_type}")
            file_save_path = self.generate_file_name(system_type)
            print(f'USING PATH: {file_save_path}')
            ids = []

            if os.path.exists(file_save_path):
                print(f'Path exists: {file_save_path}')
                existing_result = pd.read_csv(file_save_path)
                ids = existing_result['dataset_id'].values

            columns = [
                'exact_match', 'f1_score', 'time', 'step', 'dataset_id',
                'hit_rate', 'hits', 'total_retrieved', 'expected_count', 'error'
            ]

            file_is_new = not os.path.exists(file_save_path)

            with open(file_save_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                if file_is_new:
                    writer.writeheader()

                try:
                    for index, row in self.dataset.iterrows():
                        dataset_id = row[self.id_column]

                        if len(ids) > 0 and self.dataset_name == 'qasina' and int(dataset_id) in ids.astype(int):
                            print(f"Skipping row with dataset id: {dataset_id}")
                            continue
                        elif len(ids) > 0 and dataset_id in ids:
                            print(f"Skipping row with dataset id: {dataset_id}")
                            continue

                        result = {
                            'exact_match': None,
                            'f1_score': None,
                            'time': None,
                            'step': None,
                            'dataset_id': dataset_id,
                            'hit_rate': None,
                            'hits': None,
                            'total_retrieved': None,
                            'expected_count': None,
                            'error': None,
                        }

                        try:
                            if row[self.question_column] is None or row[self.answer_column] is None:
                                print(f'Skipping row because answer or question is None index: {index}')
                                result['error'] = 'Skipped: answer or question is None'
                                writer.writerow(result)
                                f.flush()
                                continue
                            elif not isinstance(row[self.answer_column], str):
                                print(f'Cleaned answer is not string: {row[self.answer_column]}')
                                result['error'] = f'Skipped: answer is not string, got {type(row[self.answer_column]).__name__}'
                                writer.writerow(result)
                                f.flush()
                                continue

                            start_time = time.time()
                            answer, retrieve_count, hit_rate_stats = self.answer_question(
                                question=row[self.question_column],
                                system_type=system_type,
                                question_id=str(dataset_id)
                            )

                            end_time = time.time()
                            elapsed = end_time - start_time

                            scores = EvaluationHelper.compute_scores(
                                a_gold=row[self.answer_column],
                                a_pred=answer
                            )

                            result['exact_match'] = scores.get('exact_match')
                            result['f1_score'] = scores.get('f1_score')
                            result['time'] = elapsed
                            result['step'] = retrieve_count
                            result['dataset_id'] = dataset_id

                            if hit_rate_stats is not None:
                                result['hit_rate'] = hit_rate_stats['hit_rate']
                                result['hits'] = hit_rate_stats['hits']
                                result['total_retrieved'] = hit_rate_stats['total_retrieved']
                                result['expected_count'] = hit_rate_stats['expected_count']

                            print(f"\n[Q] {row[self.question_column]}")
                            print(f"[A] {answer}")
                            print("[Result]:")
                            pprint(result, width=60)

                        except KeyboardInterrupt:
                            print(f'\n⚠️  Interrupted at index {index}, saving progress...')
                            writer.writerow(result)
                            f.flush()
                            print(f'Progress saved to: {file_save_path}')
                            raise
                        except Exception as e:
                            traceback.print_exc()
                            print(f'Error when trying to answer index: {index}')
                            result['error'] = str(e)

                        writer.writerow(result)
                        f.flush()

                except KeyboardInterrupt:
                    print(f'Process interrupted. Results saved to: {file_save_path}')
                    raise

            print(f"Final experiment done: {file_save_path}")
        except KeyboardInterrupt:
            print('Process stopped by user.')
        except Exception as e:
            traceback.print_exc()
            print(f'Error doing final experiment: {e}')

    def answer_question(self,
                        question: str,
                        system_type: SystemType,
                        question_id: str) -> Tuple[str, int, Optional[Dict]]:
        if system_type is SystemType.ADAPTIVE:
            classification = self.classifier.classify(text=question)
            system_type = self.type_mapping.get(classification)
            print(f'Adaptive answering using type {system_type}')
        if system_type is None:
            raise ValueError(f"Unsupported system type: {system_type}")

        retriever = self.method_map.get(system_type)

        if retriever is None:
            raise ValueError(f"Unsupported system type: {system_type}")

        result = retriever.answer(
            query=question,
            with_logging=False,
            index=self.dataset_index,
            answer=None,
            supporting_facts=[],
            question_id=question_id,
            retrieval_type=self.retrieval_type
        )

        return result

    def _set_random_seeds(self):
        """
        Set all random seeds for reproducibility.
        This ensures:
        - Dataset sampling is deterministic
        - Python's random module is seeded
        - NumPy operations are deterministic
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        print(f"✅ Random seeds set: python={self.random_seed}, numpy={self.random_seed}")

    def _log_experiment_config(self):
        """
        Log all experiment parameters to help with reproducibility.
        Saves config to JSON file alongside results.
        """
        config = {
            "model_type": self.model_type,
            "dataset_name": self.dataset_name,
            "dataset_index": self.dataset_index,
            "dataset_size": len(self.dataset),
            "n_bootstrap_samples": self.n_bootstrap_samples,
            "random_seed": self.random_seed,
            "question_column": self.question_column,
            "answer_column": self.answer_column,
            "id_column": self.id_column,
            "experiment_result_folder": self.experiment_result_folder,
            "retrieval_type": self.retrieval_type
        }

        # Create config directory if needed
        config_dir = f'{self.experiment_result_folder}/{self.dataset_name}'
        Path(config_dir).mkdir(parents=True, exist_ok=True)

        # Save to JSON
        config_path = f'{config_dir}/{self.model_type}_experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Experiment config saved to: {config_path}")

    def generate_file_name(self, system_type: SystemType, bootstrap_run: int = None) -> str:
        """
        Generate file path for experiment results.

        Args:
            system_type: Type of RAG system
            bootstrap_run: If provided, adds bootstrap run suffix
        """
        folder = f'{self.experiment_result_folder}/{self.dataset_name}'
        os.makedirs(folder, exist_ok=True)

        base_name = f'{self.model_type}_{reverse_mapping[system_type]}_{self.retrieval_type}'
        sanitized_path = re.sub(r'[^A-Za-z0-9/_]', '_', base_name)

        if bootstrap_run is not None:
            return f'{folder}/{sanitized_path}_bootstrap_run{bootstrap_run}.csv'

        return f'{folder}/{sanitized_path}.csv'

    def run_bootstrap_experiments(
        self,
        system_type: SystemType,
        n_samples: Optional[int] = None,
        resume: bool = True
    ):
        """
        Run experiments multiple times to capture LLM variability.

        Args:
            system_type: Which RAG system to test
            n_samples: Number of bootstrap iterations (default from config)
            resume: If True, skip runs that already have results

        Process:
        1. Check for existing bootstrap runs
        2. Run process() for missing iterations
        3. Save each run with run index in filename
        4. Log progress and timing
        """
        n_samples = n_samples or self.n_bootstrap_samples

        print(f"\n{'='*60}")
        print(f"🔄 Bootstrap Testing: {system_type.value}")
        print(f"   System: {self.model_type}")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Samples: {n_samples}")
        print(f"   Resume: {resume}")
        print(f"{'='*60}\n")

        # Check for existing runs if resume is True
        existing_runs = []
        corrupted_runs = []
        if resume:
            for run_idx in range(1, n_samples + 1):
                file_path = self.generate_file_name(system_type, bootstrap_run=run_idx)
                if not os.path.exists(file_path):
                    continue

                try:
                    existing_df = pd.read_csv(file_path)
                    total_rows = len(existing_df)
                    error_rows = existing_df['error'].notna().sum() + existing_df['exact_match'].isna().sum()
                    error_rows = max(error_rows, existing_df['error'].notna().sum())
                    error_rate = error_rows / total_rows if total_rows > 0 else 1.0

                    if error_rate > 0.5:
                        corrupted_runs.append(run_idx)
                        corrupted_path = f"{file_path}.corrupted"
                        if os.path.exists(corrupted_path):
                            os.remove(corrupted_path)
                        os.rename(file_path, corrupted_path)
                        print(f"⚠️  Run {run_idx} has {error_rate:.1%} error rate, marking as corrupted")
                    else:
                        existing_runs.append(run_idx)
                except Exception as e:
                    print(f"⚠️  Could not read run {run_idx} file, marking as corrupted: {e}")
                    corrupted_runs.append(run_idx)
                    corrupted_path = f"{file_path}.corrupted"
                    if os.path.exists(file_path):
                        if os.path.exists(corrupted_path):
                            os.remove(corrupted_path)
                        os.rename(file_path, corrupted_path)

            if existing_runs:
                print(f"⏭️  Found {len(existing_runs)} existing valid runs, will skip them")
                print(f"   Skipping runs: {existing_runs}")
            if corrupted_runs:
                print(f"⚠️  Found {len(corrupted_runs)} corrupted runs, will re-run them")
                print(f"   Re-running runs: {corrupted_runs}")

        # Run experiments for missing iterations
        start_time = time.time()
        completed_runs = 0

        for run_idx in range(1, n_samples + 1):
            if resume and run_idx in existing_runs:
                print(f"⏭️  Skipping run {run_idx}/{n_samples} (already exists)")
                continue

            run_start_time = time.time()
            print(f"\n🚀 Starting run {run_idx}/{n_samples} - {time.strftime('%H:%M:%S')}")

            # Run standard process
            self.process(system_type)

            # Rename file to include bootstrap run index
            old_file = self.generate_file_name(system_type)
            new_file = self.generate_file_name(system_type, bootstrap_run=run_idx)

            if os.path.exists(old_file):
                os.rename(old_file, new_file)
                print(f"✅ Saved run {run_idx} to: {new_file}")

            run_elapsed = time.time() - run_start_time
            print(f"⏱️  Run {run_idx} completed in {run_elapsed:.2f}s")

            completed_runs += 1

        total_elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"✅ Bootstrap testing complete!")
        print(f"   Total new runs: {completed_runs}")
        print(f"   Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
        print(f"{'='*60}\n")

    def cleanup_bootstrap_results(
        self,
        system_types: Optional[List[SystemType]] = None,
        n_samples: Optional[int] = None,
        error_threshold: float = 0.5,
    ):
        """
        Scan bootstrap run files and remove/rename those with high error rates.

        Args:
            system_types: Which systems to check (default: all)
            n_samples: Number of bootstrap samples (default from config)
            error_threshold: Error rate above which a file is considered corrupted (default: 0.5)
        """
        import glob

        n_samples = n_samples or self.n_bootstrap_samples
        if system_types is None:
            system_types = list(SystemType)

        reverse_mapping = {v: k for k, v in self.type_mapping.items()}
        folder = os.path.join(self.base_dir, self.experiment_result_folder, self.dataset_name)
        os.makedirs(folder, exist_ok=True)

        total_cleaned = 0

        for system_type in system_types:
            method_name = reverse_mapping.get(system_type, system_type.value)
            sanitized_method = re.sub(r'[^A-Za-z0-9]', '_', method_name)
            pattern = os.path.join(folder, f'{self.model_type}_{sanitized_method}_bootstrap_run*.csv')

            files = sorted(glob.glob(pattern))
            if not files:
                print(f"  No bootstrap files found for {method_name}")
                continue

            print(f"\n  Checking {method_name}: {len(files)} files")

            for file_path in files:
                filename = os.path.basename(file_path)
                try:
                    df = pd.read_csv(file_path)
                    total_rows = len(df)
                    if total_rows == 0:
                        print(f"    ❌ {filename}: empty file, removing")
                        os.remove(file_path)
                        total_cleaned += 1
                        continue

                    error_count = df['error'].notna().sum()
                    null_count = df['exact_match'].isna().sum()
                    error_rate = error_count / total_rows if total_rows > 0 else 1.0

                    if error_rate > error_threshold:
                        corrupted_path = f"{file_path}.corrupted"
                        if os.path.exists(corrupted_path):
                            os.remove(corrupted_path)
                        os.rename(file_path, corrupted_path)
                        print(f"    ❌ {filename}: {error_rate:.1%} errors ({error_count}/{total_rows}), renamed to .corrupted")
                        total_cleaned += 1
                    elif null_count > error_threshold * total_rows:
                        corrupted_path = f"{file_path}.corrupted"
                        if os.path.exists(corrupted_path):
                            os.remove(corrupted_path)
                        os.rename(file_path, corrupted_path)
                        print(f"    ❌ {filename}: {null_count/total_rows:.1%} null results ({null_count}/{total_rows}), renamed to .corrupted")
                        total_cleaned += 1
                    else:
                        print(f"    ✅ {filename}: OK ({error_count} errors, {null_count} nulls out of {total_rows})")
                except Exception as e:
                    print(f"    ❌ {filename}: could not read file ({e}), removing")
                    os.remove(file_path)
                    total_cleaned += 1

        print(f"\n{'='*60}")
        print(f"🧹 Cleanup complete! Removed/renamed {total_cleaned} corrupted files")
        print(f"{'='*60}\n")

    def aggregate_bootstrap_results(
        self,
        system_type: SystemType,
        n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load all bootstrap runs and compute per-question statistics.

        Args:
            system_type: Which system to aggregate
            n_samples: Number of bootstrap samples (default from config)

        Returns:
            DataFrame with per-question statistics:
            - mean, std, min, max, median for each metric
            - 95% confidence intervals (2.5th and 97.5th percentiles)
        """
        n_samples = n_samples or self.n_bootstrap_samples

        print(f"\n📊 Aggregating bootstrap results for {system_type.value}")
        print(f"   Loading {n_samples} runs...\n")

        # Load all bootstrap run files
        all_runs = []
        for run_idx in range(1, n_samples + 1):
            file_path = self.generate_file_name(system_type, bootstrap_run=run_idx)

            if not os.path.exists(file_path):
                print(f"❌ Warning: Run {run_idx} not found at {file_path}")
                continue

            try:
                df = pd.read_csv(file_path)
                df['run_id'] = run_idx
                all_runs.append(df)
                print(f"✅ Loaded run {run_idx}: {len(df)} questions")
            except Exception as e:
                print(f"❌ Error loading run {run_idx}: {e}")

        if not all_runs:
            raise ValueError(f"No bootstrap runs found for {system_type.value}")

        # Combine all runs
        combined = pd.concat(all_runs, ignore_index=True)
        print(f"\n✅ Combined {len(all_runs)} runs, total {len(combined)} observations")

        # Identify metric columns
        metric_columns = ['exact_match', 'f1_score', 'time', 'step', 'hit_rate', 'hits', 'total_retrieved', 'expected_count']
        available_metrics = [col for col in metric_columns if col in combined.columns]

        print(f"📈 Computing statistics for metrics: {available_metrics}\n")

        # Compute statistics per question
        stats_list = []

        for dataset_id in combined['dataset_id'].unique():
            subset = combined[combined['dataset_id'] == dataset_id]
            row_stats = {'dataset_id': dataset_id}

            for metric in available_metrics:
                values = subset[metric].values

                # Basic statistics
                row_stats[f'{metric}_mean'] = np.mean(values)
                row_stats[f'{metric}_std'] = np.std(values, ddof=1)
                row_stats[f'{metric}_min'] = np.min(values)
                row_stats[f'{metric}_max'] = np.max(values)
                row_stats[f'{metric}_median'] = np.median(values)

                # 95% confidence interval using percentiles
                row_stats[f'{metric}_ci_lower'] = np.percentile(values, 2.5)
                row_stats[f'{metric}_ci_upper'] = np.percentile(values, 97.5)

                # Count of valid (non-NaN) observations
                row_stats[f'{metric}_count'] = np.sum(~np.isnan(values))

            stats_list.append(row_stats)

        stats_df = pd.DataFrame(stats_list)

        # Save aggregated results
        stats_file = self.generate_bootstrap_stats_file(system_type)
        stats_df.to_csv(stats_file, index=False)
        print(f"✅ Saved bootstrap statistics to: {stats_file}\n")

        # Print summary statistics
        print(f"📊 Overall Statistics Summary:")
        for metric in available_metrics:
            mean_col = f'{metric}_mean'
            if mean_col in stats_df.columns:
                overall_mean = stats_df[mean_col].mean()
                print(f"   {metric}: {overall_mean:.4f}")

        return stats_df

    def generate_bootstrap_stats_file(self, system_type: SystemType) -> str:
        """
        Generate file path for aggregated bootstrap statistics.
        """
        base_name = f'{self.model_type}_{reverse_mapping[system_type]}_{self.retrieval_type}'
        sanitized_path = re.sub(r'[^A-Za-z0-9/_]', '_', base_name)
        folder = f'{self.experiment_result_folder}/{self.dataset_name}'
        return f'{folder}/{sanitized_path}_bootstrap_stats.csv'

    def generate_bootstrap_summary_file(self, system_type: SystemType) -> str:
        """
        Generate file path for bootstrap summary JSON.
        """
        base_name = f'{self.model_type}_{reverse_mapping[system_type]}_{self.retrieval_type}'
        sanitized_path = re.sub(r'[^A-Za-z0-9/_]', '_', base_name)
        folder = f'{self.experiment_result_folder}/{self.dataset_name}'
        return f'{folder}/{sanitized_path}_bootstrap_summary.json'

    def compute_statistical_significance(
        self,
        system_type_1: SystemType,
        system_type_2: SystemType,
        metric: str = 'exact_match',
        n_samples: Optional[int] = None,
        n_bootstrap_resamples: int = 10000,
        confidence_level: float = 0.95
    ) -> dict:
        """
        Perform bootstrap hypothesis test to compare two systems.

        Tests whether difference in mean scores between two systems
        is statistically significant.

        Args:
            system_type_1: First system to compare
            system_type_2: Second system to compare
            metric: Metric to compare ('exact_match' or 'f1_score' or 'hit_rate')
            n_samples: Number of bootstrap samples per system
            n_bootstrap_resamples: Number of resamples for bootstrap test
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)

        Returns:
            Dictionary with:
            - mean_diff: Difference in means (system_1 - system_2)
            - p_value: Two-tailed p-value
            - ci_lower: Lower bound of confidence interval
            - ci_upper: Upper bound of confidence interval
            - significant: True if p_value < 0.05
            - mean_1: Mean of system 1
            - mean_2: Mean of system 2
        """
        n_samples = n_samples or self.n_bootstrap_samples

        print(f"\n🔬 Statistical Significance Test")
        print(f"   Comparing: {reverse_mapping[system_type_1]} vs {reverse_mapping[system_type_2]}")
        print(f"   Metric: {metric}")
        print(f"   Bootstrap samples: {n_samples}")
        print(f"   Resamples: {n_bootstrap_resamples}")

        # Load aggregated results for both systems
        stats_1 = self.aggregate_bootstrap_results(system_type_1, n_samples)
        stats_2 = self.aggregate_bootstrap_results(system_type_2, n_samples)

        # Get mean scores per question
        mean_col = f'{metric}_mean'
        means_1 = stats_1[mean_col].values
        means_2 = stats_2[mean_col].values

        # Compute observed difference
        observed_diff = np.mean(means_1) - np.mean(means_2)

        print(f"\n   Mean {metric} for system 1: {np.mean(means_1):.4f}")
        print(f"   Mean {metric} for system 2: {np.mean(means_2):.4f}")
        print(f"   Observed difference: {observed_diff:.4f}")

        # Perform bootstrap test on difference of means
        def diff_stat(data1, data2):
            """Statistic: difference of means"""
            return np.mean(data1) - np.mean(data2)

        try:
            # Use scipy's bootstrap function
            res = scipy_bootstrap(
                (means_1, means_2),
                diff_stat,
                n_resamples=n_bootstrap_resamples,
                confidence_level=confidence_level,
                method='percentile'
            )

            # Get confidence interval
            ci_lower, ci_upper = res.confidence_interval

            # Compute two-tailed p-value
            # Proportion of bootstrap samples where |diff| >= |observed_diff|
            bootstrap_distribution = res.bootstrap_distribution
            p_value = np.mean(np.abs(bootstrap_distribution) >= np.abs(observed_diff))

            significant = p_value < 0.05

            print(f"\n   95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"   p-value: {p_value:.4f}")
            print(f"   Significant at p<0.05: {significant}\n")

            return {
                'system_1': reverse_mapping[system_type_1],
                'system_2': reverse_mapping[system_type_2],
                'metric': metric,
                'mean_1': np.mean(means_1),
                'mean_2': np.mean(means_2),
                'mean_diff': observed_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'significant': significant,
                'confidence_level': confidence_level,
                'n_bootstrap_resamples': n_bootstrap_resamples
            }

        except Exception as e:
            print(f"❌ Error in bootstrap test: {e}")
            # Fallback: simple permutation test if scipy fails
            print("   Fallback: Using simple permutation test")

            n_perm = 10000
            perm_diffs = []

            for _ in range(n_perm):
                # Pool data and randomly permute
                pooled = np.concatenate([means_1, means_2])
                perm = np.random.permutation(pooled)
                perm_1 = perm[:len(means_1)]
                perm_2 = perm[len(means_1):]
                perm_diffs.append(np.mean(perm_1) - np.mean(perm_2))

            perm_diffs = np.array(perm_diffs)
            p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
            ci_lower = np.percentile(perm_diffs, 2.5)
            ci_upper = np.percentile(perm_diffs, 97.5)
            significant = p_value < 0.05

            return {
                'system_1': reverse_mapping[system_type_1],
                'system_2': reverse_mapping[system_type_2],
                'metric': metric,
                'mean_1': np.mean(means_1),
                'mean_2': np.mean(means_2),
                'mean_diff': observed_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'significant': significant,
                'confidence_level': confidence_level,
                'method': 'permutation_test'
            }

    def compare_all_systems(
        self,
        metrics: List[str] = ['exact_match', 'f1_score', 'hit_rate'],
        n_samples: Optional[int] = None
    ) -> dict:
        """
        Perform pairwise statistical significance tests for all system combinations.

        Args:
            metrics: List of metrics to test (e.g., ['exact_match', 'f1_score', 'hit_rate'])
            n_samples: Number of bootstrap samples per system

        Returns:
            Nested dictionary with results for each system pair and metric
        """
        system_types = [
            SystemType.NON,
            SystemType.SINGLE,
            SystemType.MULTI,
            SystemType.ADAPTIVE
        ]

        results = {}

        print(f"\n{'='*60}")
        print(f"🔬 All-Pairs Statistical Comparison")
        print(f"   Metrics: {metrics}")
        print(f"{'='*60}\n")

        for metric in metrics:
            results[metric] = {}

            for i in range(len(system_types)):
                for j in range(i + 1, len(system_types)):
                    sys1 = system_types[i]
                    sys2 = system_types[j]

                    comparison_result = self.compute_statistical_significance(
                        sys1, sys2, metric, n_samples
                    )

                    # Use system names as keys
                    key = f"{comparison_result['system_1']}_vs_{comparison_result['system_2']}"
                    results[metric][key] = comparison_result

        # Save comparison results
        self._save_comparison_results(results)

        return results

    def _save_comparison_results(self, results: dict):
        """
        Save statistical comparison results to JSON file.
        """
        output_file = f'{self.experiment_result_folder}/{self.dataset_name}/statistical_comparisons.json'

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else o)

        print(f"✅ Saved comparison results to: {output_file}\n")

    def generate_paper_tables(
        self,
        n_samples: Optional[int] = None
    ) -> dict:
        """
        Generate tables ready for paper publication.

        Creates:
        1. Performance table with mean ± 95% CI
        2. Pairwise comparison table with p-values

        Args:
            n_samples: Number of bootstrap samples

        Returns:
            Dictionary with DataFrames for each table
        """
        n_samples = n_samples or self.n_bootstrap_samples

        print(f"\n{'='*60}")
        print(f"📄 Generating Paper Tables")
        print(f"{'='*60}\n")

        system_types = [
            SystemType.NON,
            SystemType.SINGLE,
            SystemType.MULTI,
            SystemType.ADAPTIVE
        ]

        # Aggregate results for all systems
        all_stats = {}
        for sys_type in system_types:
            try:
                all_stats[sys_type] = self.aggregate_bootstrap_results(sys_type, n_samples)
            except Exception as e:
                print(f"⚠️  Skipping {sys_type.value}: {e}")

        # Table 1: Performance with confidence intervals
        performance_data = []
        for sys_type, stats_df in all_stats.items():
            sys_name = reverse_mapping[sys_type]

            for metric in ['exact_match', 'f1_score', 'hit_rate']:
                mean_col = f'{metric}_mean'
                ci_lower_col = f'{metric}_ci_lower'
                ci_upper_col = f'{metric}_ci_upper'

                if mean_col in stats_df.columns:
                    mean = stats_df[mean_col].mean()
                    ci_lower = stats_df[ci_lower_col].mean()
                    ci_upper = stats_df[ci_upper_col].mean()

                    # Format as mean ± 95% CI
                    formatted = f"{mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"

                    performance_data.append({
                        'System': sys_name,
                        'Metric': metric.upper(),
                        'Performance': formatted,
                        'Mean': mean,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper
                    })

        performance_table = pd.DataFrame(performance_data)

        # Table 2: Pairwise comparisons with p-values
        comparisons = self.compare_all_systems(metrics=['exact_match', 'f1_score', 'hit_rate'], n_samples=n_samples)

        comparison_data = []
        for metric, comp_dict in comparisons.items():
            for comp_name, comp_result in comp_dict.items():
                # Format p-value
                p_val = comp_result['p_value']
                if p_val < 0.001:
                    p_str = "<0.001***"
                elif p_val < 0.01:
                    p_str = f"{p_val:.3f}**"
                elif p_val < 0.05:
                    p_str = f"{p_val:.3f}*"
                else:
                    p_str = f"{p_val:.3f}"

                comparison_data.append({
                    'Comparison': comp_name,
                    'Metric': metric.upper(),
                    'Mean_Diff': comp_result['mean_diff'],
                    '95%_CI': f"[{comp_result['ci_lower']:.3f}, {comp_result['ci_upper']:.3f}]",
                    'p_value': p_str,
                    'Significant': comp_result['significant']
                })

        comparison_table = pd.DataFrame(comparison_data)

        # Save tables
        tables_dir = f'{self.experiment_result_folder}/{self.dataset_name}/tables'
        Path(tables_dir).mkdir(parents=True, exist_ok=True)

        performance_file = f'{tables_dir}/performance_table.csv'
        comparison_file = f'{tables_dir}/comparison_table.csv'

        performance_table.to_csv(performance_file, index=False)
        comparison_table.to_csv(comparison_file, index=False)

        print(f"✅ Saved performance table to: {performance_file}")
        print(f"✅ Saved comparison table to: {comparison_file}\n")

        # Print tables for quick preview
        print("📊 Performance Table:")
        print(performance_table.to_string(index=False))
        print(f"\n🔬 Comparison Table:")
        print(comparison_table.to_string(index=False))

        return {
            'performance': performance_table,
            'comparison': comparison_table
        }

    def summarize_results(self, output_path: str = None) -> pd.DataFrame:
        output_path = output_path or f'{self.experiment_result_folder}/final_report.csv'

        model_display_names = {
            'gemma3_latest': 'Gemma 3',
            'qwen3_8b': 'Qwen 3',
            'stable_qwen': 'Stable Qwen',
        }

        results = []
        result_folder = Path(self.experiment_result_folder)

        if not result_folder.exists():
            raise FileNotFoundError(f"Experiment results folder not found: {result_folder}")

        for dataset_dir in sorted(result_folder.iterdir()):
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name

            for csv_file in sorted(dataset_dir.glob('*.csv')):
                if '_bootstrap' in csv_file.stem:
                    continue

                stem = csv_file.stem

                model_key = None
                method = None
                for prefix, display_name in model_display_names.items():
                    if stem.startswith(prefix):
                        model_key = prefix
                        method = stem[len(prefix) + 1:]
                        break

                if model_key is None or method is None:
                    print(f"⚠️  Skipping unrecognized file: {csv_file}")
                    continue

                retrieval_type = 'lexical'
                for suffix in ['semantic', 'hybrid']:
                    if method.endswith(f'_{suffix}'):
                        retrieval_type = suffix
                        method = method[:-(len(suffix) + 1)]
                        break

                try:
                    df = pd.read_csv(csv_file, on_bad_lines='warn')
                except Exception as e:
                    print(f"⚠️  Error reading {csv_file}: {e}")
                    continue

                total_data = len(df)

                row = {
                    'method': method,
                    'retrieval_type': retrieval_type,
                    'llm_model': model_display_names.get(model_key, model_key),
                    'dataset': dataset_name,
                    'total_data': total_data,
                    'exact_match': int(df['exact_match'].sum()),
                    'accuracy': df['exact_match'].mean(),
                    'f1_mean': df['f1_score'].mean(),
                    'step_mean': df['step'].mean(),
                    'time_mean': df['time'].mean(),
                }

                hit_rate_columns = ['hit_rate', 'hits', 'total_retrieved', 'expected_count']
                for col in hit_rate_columns:
                    if col in df.columns and df[col].notna().any():
                        row[f'{col}_mean'] = df[col].mean()
                    else:
                        row[f'{col}_mean'] = None

                results.append(row)

        if not results:
            raise ValueError("No result files found to summarize.")

        report_df = pd.DataFrame(results)
        report_df.to_csv(output_path, index=False)

        print(f"✅ Summary report saved to: {output_path}")
        print(f"   Total rows: {len(report_df)}")

        return report_df

