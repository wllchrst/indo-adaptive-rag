import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ClassifierConfig:
    # Model settings
    model_path: str = 'indobenchmark/indobert-base-p1'
    num_labels: int = 3
    
    # Data settings
    undersample: bool = True
    use_balanced_only: bool = False
    train_size: float = 0.8
    val_size: float = 0.2
    test_size: float = 0.2
    max_length: int = 512
    file_path: Optional[str] = None
    
    # Reproducibility settings
    random_seed: int = 42
    torch_seed: int = 42
    numpy_seed: int = 42
    
    # Training hyperparameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.3
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # Model architecture
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Output paths
    output_dir: str = 'saved_model'
    confusion_matrix_dir: str = 'confusion_matrixes'
    
    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    @classmethod
    def load(cls, path: str) -> 'ClassifierConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
