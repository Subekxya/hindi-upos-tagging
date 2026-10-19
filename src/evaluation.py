import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_token_predictions(predictions, labels):
    """Calculate accuracy, precision, recall, and F1 for UPOS tagging"""
    pred_ids = np.argmax(predictions, axis=2)

    y_true = []
    y_pred = []

    for pred_sentence, true_sentence in zip(pred_ids, labels):
        for pred_id, true_id in zip(pred_sentence, true_sentence):
            if true_id != -100:
                y_true.append(true_id)
                y_pred.append(pred_id)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1}