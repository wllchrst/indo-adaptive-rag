import torch
import evaluate
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from training_classifier.data_loader import DataLoader
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
from typing import Tuple, Optional

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }


class TrainClassifier:
    def __init__(self, undersample: bool, file_path: Optional[str] = None,
                 model_path: str = 'indobenchmark/indobert-base-p1'):
        self.data_loader = DataLoader(undersample=undersample, file_path=file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.label2id = {"A": 0, "B": 1, "C": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.class_names = [self.id2label[i] for i in range(len(self.label2id))]
        self.max_length = 512

        self.train_dataset, self.val_dataset, self.test_dataset = \
            self.process_dataset(dataset=self.data_loader.dataset)

    def process_dataset(
            self,
            dataset: Dataset,
            train_size: float = 0.8,
            val_size: float = 0.2,
            test_size: float = 0.2,
            seed: int = 42,
    ) -> Tuple[Dataset, Dataset, Dataset]:

        # add numeric labels
        def encode_labels(example):
            example["label"] = self.label2id[example["classification"]]
            return example

        dataset = dataset.map(encode_labels)

        # tokenize
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

    def train_model(self,
                    training_dataset: Dataset,
                    validation_dataset: Dataset,
                    testing_dataset: Dataset,
                    model_path: str,
                    output_dir='saved_model',
                    output_confusion_matrix='confusion_matrixes'):
        torch.cuda.empty_cache()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3
        )

        model.config.hidden_dropout_prob = 0.1
        model.config.attention_probs_dropout_prob = 0.1

        if torch.cuda.is_available():
            print(f"Training model is using GPU {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
            model.to(device)
        else:
            print(f'Training model is using CPU')

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.3,
            logging_dir="./logs",
            logging_steps=10,
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            save_total_limit=1,
            report_to="none",
            learning_rate=2e-5,
            gradient_accumulation_steps=2,
            max_grad_norm=1
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=training_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        evaluation_result = trainer.evaluate(testing_dataset)
        print(f'Evaluation result: {evaluation_result}')

        model_name = model_path.replace("/", "_")
        model_save_path = os.path.join(output_dir, model_name)
        confusion_save_path = os.path.join(output_confusion_matrix, f'{model_name}.jpg')

        trainer.save_model(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)

        predictions = trainer.predict(testing_dataset)

        self.generate_confusion_matrix(
            eval_pred=(predictions.predictions, predictions.label_ids),
            labels=list(self.label2id.values()),
            class_names=self.class_names,
            save_path=confusion_save_path
        )

    def generate_confusion_matrix(
            self,
            eval_pred,
            labels: list[int],
            save_path: str,
            class_names=None,
    ):
        """
        Generates and saves the confusion matrix as a .jpg file.

        Args:
        - eval_pred: Tuple containing (logits, labels).
        - labels: List of unique labels.
        - class_names: List of class names (optional).
        - save_path: File path to save the confusion matrix image.
        """
        logits, true_labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        cm = confusion_matrix(true_labels, predictions, labels=labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, format="jpg", dpi=300)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
