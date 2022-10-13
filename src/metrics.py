
from typing import Tuple

from numpy import array
from sklearn.metrics import precision_score, recall_score


def eval_metrics(
        y_true: array,
        y_pred: array
) -> Tuple[float, float]:
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return precision, recall
