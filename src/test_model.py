
import pandas as pd

from configs.data_conf import data_config
from metrics import eval_metrics


def main(data_config: dict, model):
    test_data = pd.read_csv(data_config['test_path'])
    data_cols = ['name_1', 'name_2']
    y_pred = model(test_data[data_cols].values)
    y_true = test_data['is_duplicate']
    precision, recall = eval_metrics(y_true, y_pred)
    return precision, recall


if __name__ == '__main__':
    model = 'model'
    main(data_config, model)
