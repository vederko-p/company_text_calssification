import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from configs.data_conf import data_config
from configs.preprocess_conf import preprocess_config


def tokenizer():

    df = pd.read_csv(data_config['train_path'])
    words = ''
    for name_1, name_2 in df.values:
        words = words + name_1 + ' ' + name_2 + ' '
    tokenizer = Tokenizer(num_words=100000, filters='–—!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\xa0–\ufeff',
                          lower=True, split=' ', char_level=False, oov_token = 'unknown')
    tokenizer.fit_on_texts([words])

    with open(preprocess_config['output_tokenizer_path'], "wb") as f:
        pickle.dump(tokenizer, f)

    print('Tokenizer save')
    pass
