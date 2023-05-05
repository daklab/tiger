import tensorflow as tf
from tensorflow.keras.layers import Layer


class SequenceSequentialWithNonSequenceBypass(Layer):
    def __init__(self, input_parser: Layer, sequence_layers: [Layer]):
        super().__init__()
        self.input_parser = input_parser
        self.sequence_layers = sequence_layers

    def call(self, x, **kwargs):
        *sequence_representation, non_sequence_features = self.input_parser.call(x)
        for layer in self.sequence_layers:
            if isinstance(sequence_representation, (list, tuple)):
                sequence_representation = layer(*sequence_representation, **kwargs)
            else:
                sequence_representation = layer(sequence_representation, **kwargs)
        return tf.concat([sequence_representation, non_sequence_features], axis=-1)


class OneHotInputParser(Layer):
    def __init__(self, target_len: int, context_5p: int, context_3p: int, use_guide_seq: bool, pad_guide_seq: bool):
        super().__init__()
        self.target_len = context_5p + target_len + context_3p
        self.guide_len = (self.target_len if pad_guide_seq else target_len) if use_guide_seq else 0

    def call(self, x):
        target_one_hot = x[:, : 4 * self.target_len]
        target_one_hot = tf.reshape(target_one_hot, [tf.shape(x)[0], self.target_len, 4])
        guide_one_hot = x[:, 4 * self.target_len: 4 * (self.target_len + self.guide_len)]
        guide_one_hot = tf.reshape(guide_one_hot, [tf.shape(x)[0], self.guide_len, 4])
        non_sequence_features = x[:, 4 * (self.target_len + self.guide_len):]

        return target_one_hot, guide_one_hot, non_sequence_features


class AlignOneHotEncoding1D(Layer):
    def __init__(self,  use_guide_seq: bool):
        super().__init__()
        self.use_guide_seq = use_guide_seq

    def call(self, target_one_hot, guide_one_hot):

        # all done if model uses target sequence only
        if not self.use_guide_seq:
            return target_one_hot

        # otherwise, align and concatenate guide sequence
        else:
            tf.assert_equal(tf.shape(target_one_hot)[1], tf.shape(guide_one_hot)[1])
            return tf.concat([target_one_hot, guide_one_hot], axis=-1)


class AlignOneHotEncoding2D(Layer):
    def __init__(self,  use_guide_seq: bool):
        super().__init__()
        self.use_guide_seq = use_guide_seq

    def call(self, target_one_hot, guide_one_hot):

        # all done if model uses target sequence only
        x = tf.expand_dims(target_one_hot, axis=-1)

        # otherwise, align and concatenate guide sequence
        if self.use_guide_seq:
            tf.assert_equal(tf.shape(target_one_hot)[1], tf.shape(guide_one_hot)[1])
            x = tf.stack([target_one_hot, guide_one_hot], axis=-1)

        return tf.transpose(x, [0, 2, 1, 3])
