from predict_company import get_predict
import pandas as pd
from configs.data_conf import data_config

def main(name):
    list_company = get_predict(name)

    return list_company


if __name__ == '__main__':
    print(main())
