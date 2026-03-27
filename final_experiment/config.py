from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Config:
    classifier_model_path: str
    dataset_path: str
    dataset_index: str
    dataset_name: str
    dataset_part: float
    keep_column: List[str]
    model_type: str
    question_column: str
    answer_column: str
    id_column: str
    experiment_result_folder: str
    n_bootstrap_samples: int = 10
    random_seed: int = 42


configs: Dict[str, Config] = {
    'indoqa': Config(
        classifier_model_path='',
        dataset_path='classification_result/stable-qwen/indoqa__train.csv',
        dataset_index='indoqa',
        dataset_name='indo_qa',
        dataset_part=1,
        keep_column=['id', 'question', 'answer'],
        model_type='',
        question_column='question',
        answer_column='answer',
        id_column='id',
        experiment_result_folder='experiment_results',
        n_bootstrap_samples=10,
        random_seed=42
    ),
    'qasina': Config(
        classifier_model_path='',
        dataset_path='classification_result/stable-qwen/qasina__full.csv',
        dataset_index='qasina',
        dataset_name='qasina',
        dataset_part=1,
        keep_column=['ID', 'question', 'answer'],
        model_type='',
        question_column='question',
        answer_column='answer',
        id_column='ID',
        experiment_result_folder='experiment_results',
        n_bootstrap_samples=10,
        random_seed=42
    ),
    'hotpot': Config(
        classifier_model_path='',
        dataset_path='hotpot/validation.csv',
        dataset_index='hotpot',
        dataset_name='hotpot',
        dataset_part=1,
        keep_column=['id', 'question', 'answer'],
        model_type='',
        question_column='question',
        answer_column='answer',
        id_column='id',
        experiment_result_folder='experiment_results',
        n_bootstrap_samples=10,
        random_seed=42
    )
}
