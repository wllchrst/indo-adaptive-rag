import torch
import evaluate
import numpy as np
from training_classifier.data_loader import DataLoader
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback
from datasets import Dataset

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
    def __init__(self):
        self.data_loader = DataLoader()

    def train_model(self,
                    training_dataset: Dataset,
                    validation_dataset: Dataset,
                    testing_dataset: Dataset,
                    model_path: str):
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
            save_strategy="epoch",
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
