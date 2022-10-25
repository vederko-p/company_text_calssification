import keras.backend as K
import tensorflow as tf


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


@tf.function
def cosine_vectorized(vec):
    array1, array2 = vec
    sum_yy = K.sum(array2**2, axis=1, keepdims=1)
    sum_xx = K.sum(array1**2, axis=1, keepdims=1)
    sum_xy = K.dot(array1, K.transpose(array2))
    sum_xy = tf.linalg.diag_part(sum_xy)
    sum_xy = tf.reshape(sum_xy, [-1, 1])
    return K.abs((sum_xy/K.sqrt(sum_xx))/K.sqrt(sum_yy))
