import pickle
import numpy as np
import pandas as pd
import predict_company
from preprocessing.filter_companys import frequency_filter

from configs.preprocess_conf import preprocess_config


def get_list_company(target_company):

    with open(preprocess_config['dictionary'], "rb") as f:
        dictionary = pickle.load(f)
    with open(preprocess_config['list_company'], "rb") as f:
        list_companys = pickle.load(f)
    with open(preprocess_config['tokenizer'], "rb") as f:
        tokenizer = pickle.load(f)

    filtered_companys = frequency_filter(target_company, list_companys, dictionary)

    target_company = target_company.lower().split()
    if len(target_company) > 8:
        target_company_crop = np.array(target_company[:8])
    else:
        target_company_crop = np.zeros(8)
        target_company_crop[:len(target_company)] = target_company

    filtered_companys_crop = np.zeros((len(filtered_companys), 8))
    for i, name in enumerate(filtered_companys):
        if len(name) > 8:
            name = name[:8]
            filtered_companys_crop[i, :len(name)] = name
        else:
            filtered_companys_crop[i, :len(name)] = name

    target_token = tokenizer.texts_to_sequences(target_company_crop)
    another_token = tokenizer.texts_to_sequences(filtered_companys_crop)

    similar_company = predict_company.get_predict(target_token, another_token)

    return similar_company
