import pickle
import numpy as np
import pandas as pd
import models.siamese_network as sn
from configs.model_conf import model_config
from preprocessing.filter_companys import frequency_filter
from configs.preprocess_conf import preprocess_config


def get_predict(target_company) -> list[tuple[str, float]]:

    with open(preprocess_config['dictionary'], "rb") as f:
        dictionary = pickle.load(f)
    with open(preprocess_config['list_companys'], "rb") as f:
        list_companys = pickle.load(f)
    with open(preprocess_config['tokenizer'], "rb") as f:
        tokenizer = pickle.load(f)

    filtered_companys = frequency_filter(target_company, list_companys, dictionary)
    if not filtered_companys:
        filtered_companys = np.random.choice(list_companys, size=10)

    target_token = tokenizer.texts_to_sequences([target_company])[0]
    filtered_token = tokenizer.texts_to_sequences(filtered_companys)
    if len(target_token) > 8:
        target_token_crop = np.array([target_token[:8]])
    else:
        target_token_crop = np.zeros((1, 8))
        target_token_crop[0, :len(target_token)] = target_token
    target_token_crop = target_token_crop.repeat(len(filtered_token), axis=0)

    filtered_token_crop = np.zeros((len(filtered_token), 8))
    for i, name in enumerate(filtered_token):
        if len(name) > 8:
            name = name[:8]
            filtered_token_crop[i, :len(name)] = name
        else:
            filtered_token_crop[i, :len(name)] = name

    model = sn.get_siamese_model(8, 100, 128)
    model.load_weights(model_config['weights'])
    similarity = model.predict([target_token_crop, filtered_token_crop])

    return sorted(list(zip(filtered_companys, similarity)), key=lambda tup: tup[1], reverse=True)
