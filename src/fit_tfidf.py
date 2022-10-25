import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from configs.data_conf import data_config
from configs.preprocess_conf import preprocess_config


def tf_idf():
    df = pd.read_csv(data_config['train_path'])
    df_for_tfidf = np.hstack([df.name_1.values, df.name_2.values])
    tfv = TfidfVectorizer(min_df=5, max_df=2000)
    tfv.fit(df_for_tfidf)
    dict_words = tfv.vocabulary_
    key_words = dict_words.keys()

    with open(preprocess_config['output_tfidf_path'], "wb") as f:
        pickle.dump(list(key_words), f)

    print('Dictionary save')
    pass
