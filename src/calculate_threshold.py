import pandas as pd
import numpy as np
from typing import List
from metrics import *
from configs.data_conf import data_config
from predict_company import get_predict


test_data = pd.read_csv(pd.read_csv(data_config['test_path']))
phraze_classes_test = pd.read_csv(pd.read_csv(data_config['phraze_classes_test.csv']))
phraze_classes_test.index = phraze_classes_test['phraze']
PHRAZE_CLASSES_TEST = phraze_classes_test


class RankingPrecision:
    def __init__(self, phraze_classes_df):
        self.phraze_classes_df = phraze_classes_df

    @staticmethod
    def _evaluate_single_precision(
            class_q: set,
            smoothed_pred: List[Tuple[str, float]]
    ) -> float:
        pred_vs_true = y_true_func(class_q, smoothed_pred)
        q_class_n = len(class_q)
        precision = ranking_precision(q_class_n, pred_vs_true)
        return precision

    def __call__(
            self,
            q: str,
            p: Union[float, List[float]],
            list_of_preds: List[Tuple[str, float]]
    ) -> float:
        """Evaluate ranking precision.

        Parameters
        ----------
        p : float | List[floats]
            Threshold.
        q : str
            Phraze (query).
        phraze_classes_df : pd.DataFrame
            Data frame of (phraze, class), where phraze is index.

        Return
        ------
        ranking_precision : float
            Ranking precision.
        """

        class_q = get_phraze_class(q, self.phraze_classes_df) - {q}
        q_class_n = len(class_q)

        if not class_q:
            if isinstance(p, float):
                smoothed_pred = smooth_by_threshold(p, list_of_preds)
                precision = fix_precision_in_empty_case(smoothed_pred)
            else:
                smoothed_pred = [
                    smooth_by_threshold(pi, list_of_preds)
                    for pi in p
                ]
                precision = [
                    fix_precision_in_empty_case(sp)
                    for sp in smoothed_pred
                ]

        else:
            completed_pred = complete_predict(q_class_n, list_of_preds)
            if isinstance(p, float):
                smoothed_pred = smooth_by_threshold(p, completed_pred)
                precision = self._evaluate_single_precision(
                    class_q, smoothed_pred
                )
            else:
                smoothed_pred = [
                    smooth_by_threshold(pi, completed_pred)
                    for pi in p
                ]
                precision = [
                    self._evaluate_single_precision(class_q, sp)
                    for sp in smoothed_pred
                ]
        return precision


test_data = np.hstack([test_data.name_1.values, test_data.name_2.values])
rank_prec = RankingPrecision(PHRAZE_CLASSES_TEST)
for target in test_data:
    test_pred = get_predict(target)
    rank_prec(target, 0.5, test_pred)


