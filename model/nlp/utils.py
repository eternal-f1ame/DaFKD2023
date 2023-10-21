import math
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Tuple, List, Dict
from datasets import load_metric
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

metric = load_metric("seqeval")


class CustomAttention(nn.Module):
    def __init__(self, in_features: int, embed_dim: int = 128, activation=nn.LeakyReLU(0.2)):
        super(CustomAttention, self).__init__()
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)
        self.linear_q = nn.Linear(in_features, embed_dim)
        self.linear_k = nn.Linear(in_features, embed_dim)

    def forward(self, query, key, value, **kwargs):
        Q = self.linear_q(query)
        K = self.linear_k(query)
        dk = Q.size()[-1]
        scores = Q.bmm(K.transpose(-2, -1)) / math.sqrt(dk)
        attention = self.softmax(scores)
        return self.activation(attention.bmm(value))


def compute_ner_metrics(
    predictions: torch.Tensor, labels: List[List[int]], label_names: List[str]
) -> Dict:
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_names[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results


def compute_clf_metrics(predictions: torch.Tensor, labels: List[int], label_names: List[str]) -> Dict:
    predictions = np.argmax(predictions, axis=1)
    overall_accuracy = accuracy_score(labels, predictions)
    overall_fscore = f1_score(labels, predictions, average="macro")
    overall_precision = precision_score(labels, predictions, average="macro")
    overall_recall = recall_score(labels, predictions, average="macro")
    detailed_metrics = classification_report(
        labels, predictions, target_names=label_names, output_dict=True
    )
    return dict(
        overall_accuracy=overall_accuracy,
        overall_f1=overall_fscore,
        overall_precision=overall_precision,
        overall_recall=overall_recall,
        detailed_metrics=detailed_metrics,
    )


class compute_multi_label_metrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, predictions: torch.Tensor, labels: List[int], label_names: List[str]) -> Dict:
        predictions = sigmoid(predictions) > self.threshold
        overall_accuracy = self.hamming_score(predictions, labels)
        detailed_metrics = {}
        for i, name in enumerate(label_names):
            fscore = f1_score(predictions[:, i], labels[:, i])
            precision = precision_score(predictions[:, i], labels[:, i])
            recall = recall_score(predictions[:, i], labels[:, i])
            detailed_metrics[name] = dict(f1=fscore, precision=precision, recall=recall)
        return dict(
            overall_accuracy=overall_accuracy,
            overall_f1=np.mean([v["f1"] for v in detailed_metrics.values()]),
            overall_precision=np.mean([v["precision"] for v in detailed_metrics.values()]),
            overall_recall=np.mean([v["recall"] for v in detailed_metrics.values()]),
            detailed_metrics=detailed_metrics,
        )

    @staticmethod
    def hamming_score(y_true, y_pred):
        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(np.where(y_true[i])[0])
            set_pred = set(np.where(y_pred[i])[0])
            tmp_a = None
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
            acc_list.append(tmp_a)
        return np.mean(acc_list)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@dataclass()
class ClassifierOutput(OrderedDict):
    loss: Optional[torch.FloatTensor] = None
    fake_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    fake_logits: torch.FloatTensor = None
    probs: torch.FloatTensor = None
    fake_probs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

    def __repr__(self):
        kws = [f"{key}={type(value).__name__}" for key, value in self.__dict__.items()]
        return f"{type(self).__name__}({', '.join(kws)})"

    def __str__(self):
        return self.__repr__()
