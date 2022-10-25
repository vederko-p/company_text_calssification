import os
import pandas as pd
from typing import List, Tuple, Union


def get_phraze_class(
        q: str,
        phraze_classes_df: pd.DataFrame
) -> set:
    """Get all phrazes of the same class as given phraze.

    Parameters
    ----------
    q : str
        Phraze (query).
    phraze_classes_df : pd.DataFrame
        Data frame of (phraze, class), where phraze is index.

    Return
    ------
    class_phrazes : set
        Unique phrazes of the same class as given phraze.
    """

    target_calss = phraze_classes_df.loc[q]['class']
    class_phrazes = set(phraze_classes_df[
                            phraze_classes_df['class'] == target_calss].index)
    return class_phrazes


def complete_predict(
        n: int,
        ordered_list_of_pred: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """Slice by first n and complete predict with zeros.

    Parameters
    ----------
    n : int
        Amount of phrazes in target class.
    ordered_list_of_pred : List[Tuple[str, float]]
        List of ordered predictions from model.

    Return
    ------
    completed : List[Tuple[str, float]]
        Completed predict."""

    sliced = ordered_list_of_pred[:n]
    completed = sliced + [('', 0)] * (n - len(sliced))
    return completed


def smooth_by_threshold(
        p: float,
        list_of_pred: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """Smooth predicitons probabilty by given threshold.

    Parameters
    ----------
    p : float
        Threshold.
    list_of_pred : List[Tuple[str, float]]
        List of predictions from model.

    Return
    ------
    smoothed : List[Tuple[str, float]]
        Smoothed list of predicitons.
    """

    smoothed = [(phr, int(prob > p)) for phr, prob in list_of_pred]
    return smoothed


def y_true_func(
        true_phrazes: set,
        list_of_pred: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    """Check if predicted phrazes invloved in true ones.

    Parameters
    ----------
    true_phrazes : set
        Set of true phrazes.
    list_of_pred : List[Tuple[str, float]]
        List of predictions from model.

    Return
    ------
    checked : List[Tuple[str, float]]
        Checked list of predicitons.
    """

    checked = [
        (phr, int(phr in true_phrazes) * prob) for phr, prob in list_of_pred
    ]
    return checked


def ranking_precision(
    n: int,
    list_of_pred: List[Tuple[str, float]]
) -> float:
    precision = sum([prob for _, prob in list_of_pred]) / n
    return precision


def fix_precision_in_empty_case(
    list_of_pred: List[Tuple[str, float]]
):
    err = sum([prob for _, prob in list_of_pred])
    fixed_precision = 1 / (1 + err)
    return fixed_precision
