import tensorflow as tf
from keras import backend
from keras.losses import LossFunctionWrapper
from keras.utils import losses_utils
from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.losses.log_cosh', 'keras.losses.logcosh',
              'keras.metrics.log_cosh', 'keras.metrics.logcosh')
@tf.__internal__.dispatch.add_dispatch_support
def log_cosh(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    def _log_cosh(x):
        return x + tf.math.softplus(-2. * x) - tf.cast(tf.math.log(2.), x.dtype)

    return backend.mean(_log_cosh(y_pred - y_true), axis=-1)


class LogCosh(LossFunctionWrapper):
    def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='log_cosh'):
        super().__init__(log_cosh, name=name, reduction=reduction)
