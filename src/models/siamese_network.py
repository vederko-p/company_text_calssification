from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add
from tensorflow.keras.layers import Activation, Lambda, Embedding, SpatialDropout1D, Flatten
from tensorflow.keras.models import Model
from keras import regularizers
from models.cosine_layer import cosine_vectorized, cos_dist_output_shape


def get_siamese_model(input_shape, embedding_len, dense_units):
    def siamese_block(input_vec, embedding_len, dense_units):
        x = Embedding(100000, embedding_len, input_length=input_shape)(input_vec)
        x = SpatialDropout1D(0.2)(x)
        x = Flatten()(x)
        x = Dense(dense_units)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='linear', kernel_regularizer=regularizers.l2(0.03))(x)
        return x

    input_1 = Input(input_shape)
    x1 = siamese_block(input_1, embedding_len, dense_units)

    input_2 = Input(input_shape)
    x2 = siamese_block(input_2, embedding_len, dense_units)

    out = Lambda(cosine_vectorized, output_shape=cos_dist_output_shape)([x1, x2])
    model = Model(inputs=(input_1, input_2), outputs=out)

    return model
