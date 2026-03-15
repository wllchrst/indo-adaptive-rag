import numpy as np
import evaluate

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred, label_names):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Macro-averaged metrics (existing)
    metrics = {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision_macro": precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "recall_macro": recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"],
        "f1_macro": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
    }
    
    # Per-class metrics (NEW - addresses Priority #10)
    precision_per_class = precision_metric.compute(predictions=predictions, references=labels, average=None)["precision"]
    recall_per_class = recall_metric.compute(predictions=predictions, references=labels, average=None)["recall"]
    f1_per_class = f1_metric.compute(predictions=predictions, references=labels, average=None)["f1"]
    
    for i, label in enumerate(label_names):
        metrics[f"precision_class_{label}"] = precision_per_class[i]
        metrics[f"recall_class_{label}"] = recall_per_class[i]
        metrics[f"f1_class_{label}"] = f1_per_class[i]
    
    return metrics
