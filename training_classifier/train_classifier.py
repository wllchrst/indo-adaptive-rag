import torch
import numpy as np
import os
import json
import random
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from training_classifier.data_loader import DataLoader
from training_classifier.config import ClassifierConfig
from training_classifier.metrics_utils import compute_metrics
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from typing import Tuple, Optional


class TrainClassifier:
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._set_random_seeds()
        
        print("Initializing classifier with config:")
        print(f"  Model: {config.model_path}")
        print(f"  Undersample: {config.undersample}")
        print(f"  Random seed: {config.random_seed}")
        
        # Load data
        self.data_loader = DataLoader(undersample=config.undersample, 
                                      file_path=config.file_path)
        
        # Model setup
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.label2id = {"A": 0, "B": 1, "C": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.class_names = [self.id2label[i] for i in range(len(self.label2id))]
        self.max_length = config.max_length
        
        # Process balanced dataset
        self.train_dataset_balanced, self.val_dataset_balanced, self.test_dataset_balanced = \
            self.process_dataset(dataset=self.data_loader.dataset, seed=config.random_seed)
        
        # Process original (imbalanced) dataset for comparison (only if use_balanced_only=False)
        if not self.config.use_balanced_only:
            print("\nLoading original (imbalanced) dataset for comparison...")
            data_loader_original = DataLoader(undersample=False, file_path=config.file_path)
            _, _, self.test_dataset_original = \
                self.process_dataset(dataset=data_loader_original.dataset, seed=config.random_seed)
        
        # Save distributions
        self._save_class_distributions()
    
    def _set_random_seeds(self):
        """Set all random seeds for reproducibility (Priority #7)"""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.numpy_seed)
        torch.manual_seed(self.config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.torch_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print("All random seeds set for reproducibility")
    
    def process_dataset(
            self,
            dataset: Dataset,
            train_size: float = None,
            val_size: float = None,
            test_size: float = None,
            seed: int = None,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        
        if train_size is None:
            train_size = self.config.train_size
        if val_size is None:
            val_size = self.config.val_size
        if test_size is None:
            test_size = self.config.test_size
        if seed is None:
            seed = self.config.random_seed
        
        def encode_labels(example):
            example["label"] = self.label2id[example["classification"]]
            return example
        
        dataset = dataset.map(encode_labels)
        
        def tokenize_function(example):
            return self.tokenizer(
                example["question"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
        
        dataset = dataset.map(tokenize_function, batched=True)
        
        train_val_dataset, test_dataset = dataset.train_test_split(
            test_size=test_size, seed=seed
        ).values()
        
        train_dataset, val_dataset = train_val_dataset.train_test_split(
            test_size=val_size / (train_size + val_size),
            seed=seed
        ).values()
        
        return train_dataset, val_dataset, test_dataset
    
    def train_and_evaluate(self):
        """Main training and evaluation pipeline"""
        model_name = self.config.model_path.replace("/", "_")
        model_save_path = os.path.join(self.config.output_dir, model_name)
        os.makedirs(model_save_path, exist_ok=True)
        
        # Save config for reproducibility
        config_path = os.path.join(model_save_path, "config.json")
        self.config.save(config_path)
        print(f"Config saved to {config_path}")
        
        # Train on balanced dataset
        print("\n" + "="*60)
        print("TRAINING ON BALANCED DATASET")
        print("="*60)
        model, trainer = self._train_model(
            training_dataset=self.train_dataset_balanced,
            validation_dataset=self.val_dataset_balanced,
            model_save_path=model_save_path
        )
        
        # Evaluate on balanced test set
        print("\n" + "="*60)
        print("EVALUATING ON BALANCED TEST SET")
        print("="*60)
        eval_balanced = trainer.evaluate(self.test_dataset_balanced)
        self._save_evaluation_results(eval_balanced, model_save_path, "evaluation_balanced.json")
        self._generate_confusion_matrix(
            trainer.predict(self.test_dataset_balanced), 
            model_save_path, 
            "confusion_matrix_balanced.jpg"
        )
        
        # Evaluate on original (imbalanced) test set (only if use_balanced_only=False)
        if not self.config.use_balanced_only:
            print("\n" + "="*60)
            print("EVALUATING ON ORIGINAL (IMBALANCED) TEST SET")
            print("="*60)
            eval_original = trainer.evaluate(self.test_dataset_original)
            self._save_evaluation_results(eval_original, model_save_path, "evaluation_original.json")
            self._generate_confusion_matrix(
                trainer.predict(self.test_dataset_original), 
                model_save_path, 
                "confusion_matrix_original.jpg"
            )
        
        # Save splits info (NEW - addresses transparency requirement)
        self._save_splits_info(model_save_path)
        
        print(f"\n" + "="*60)
        print("ALL RESULTS SAVED TO:")
        print(f"  {model_save_path}")
        print("="*60)
        
        return model
    
    def _train_model(self, training_dataset, validation_dataset, model_save_path):
        """Train the model with validation"""
        torch.cuda.empty_cache()
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_path, num_labels=self.config.num_labels
        )
        
        model.config.hidden_dropout_prob = self.config.hidden_dropout_prob
        model.config.attention_probs_dropout_prob = self.config.attention_probs_dropout_prob
        
        if torch.cuda.is_available():
            print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        else:
            print("Training on CPU")
            device = torch.device("cpu")
        model.to(device)
        
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir="./logs",
            logging_steps=10,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            learning_rate=self.config.learning_rate,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, self.class_names),
        )
        
        trainer.train()
        trainer.save_model(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        
        return model, trainer
    
    def _save_class_distributions(self):
        """Save class distribution visualizations (NEW - addresses transparency)"""
        print("\nSaving class distribution visualizations...")
        
        # Balanced distribution
        df_balanced = self.train_dataset_balanced.to_pandas()
        dist_balanced = df_balanced['label'].value_counts().sort_index()
        self._plot_distribution(
            dist_balanced, 
            "Balanced Dataset Distribution",
            f"{self.config.output_dir}/balanced_dataset_distribution.png"
        )
        
        # Original distribution (only if use_balanced_only=False)
        if not self.config.use_balanced_only:
            df_original = self.test_dataset_original.to_pandas()
            dist_original = df_original['label'].value_counts().sort_index()
            self._plot_distribution(
                dist_original, 
                "Original Dataset Distribution",
                f"{self.config.output_dir}/original_dataset_distribution.png"
            )
    
    def _plot_distribution(self, distribution, title, save_path):
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(
            x=[self.id2label[i] for i in distribution.index], 
            y=distribution.values, 
            palette="viridis"
        )
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(title)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    def _save_splits_info(self, save_path):
        """Save dataset split information (NEW - addresses documentation requirement)"""
        print("\nSaving dataset splits information...")
        
        splits_info = {
            "train_balanced": len(self.train_dataset_balanced),
            "val_balanced": len(self.val_dataset_balanced),
            "test_balanced": len(self.test_dataset_balanced),
            "train_val_split_ratio": f"{self.config.train_size}:{self.config.val_size}",
            "test_split_ratio": self.config.test_size,
            "random_seed": self.config.random_seed,
        }
        
        # Add original test set info only if use_balanced_only=False
        if not self.config.use_balanced_only:
            splits_info["test_original"] = len(self.test_dataset_original)
        
        # Per-class counts
        splits_data = [
            ("train_balanced", self.train_dataset_balanced),
            ("val_balanced", self.val_dataset_balanced),
            ("test_balanced", self.test_dataset_balanced),
        ]
        
        if not self.config.use_balanced_only:
            splits_data.append(("test_original", self.test_dataset_original))
        
        for split_name, dataset in splits_data:
            df = dataset.to_pandas()
            for label in range(3):
                class_name = self.id2label[label]
                count = int((df['label'] == label).sum())
                splits_info[f"{split_name}_class_{class_name}"] = count
        
        splits_path = os.path.join(save_path, "splits_info.json")
        with open(splits_path, "w") as f:
            json.dump(splits_info, f, indent=4)
        print(f"  Saved: {splits_path}")
    
    def _save_evaluation_results(self, results, save_path, filename):
        """Save evaluation results to JSON"""
        filepath = os.path.join(save_path, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  Saved: {filepath}")
    
    def _generate_confusion_matrix(self, predictions, save_path, filename):
        """Generate and save confusion matrix with normalized version (ENHANCED)"""
        logits, true_labels = predictions.predictions, predictions.label_ids
        preds = np.argmax(logits, axis=-1)
        
        cm = confusion_matrix(true_labels, preds, labels=list(self.label2id.values()))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        ax1.set_title("Confusion Matrix (Counts)")
        
        # Normalized (NEW - shows proportions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        ax2.set_title("Confusion Matrix (Normalized)")
        
        plt.tight_layout()
        filepath = os.path.join(save_path, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filepath}")
