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


class TargetSequenceAndPositionalFeatures(Layer):
    def __init__(self, target_len: int, context_5p: int, context_3p: int, feature_channels: int):
        super().__init__()
        self.target_len = context_5p + target_len + context_3p
        self.feature_channels = feature_channels

    def call(self, x):

        # initialize index
        index = 0

        # one-hot target sequence
        num_elements = 4 * self.target_len
        target_one_hot = x[:, index: index + num_elements]
        target_one_hot = tf.reshape(target_one_hot, [tf.shape(x)[0], self.target_len, 4])
        index += num_elements

        # per-nucleotide target features
        num_elements = self.feature_channels * self.target_len
        target_features = x[:, index: index + num_elements]
        target_features = tf.reshape(target_features, [tf.shape(x)[0], self.target_len, self.feature_channels])
        index += num_elements

        # make sure everything expected was utilized
        tf.assert_equal(index, tf.shape(x)[1])

        # no non-sequence features
        non_sequence_features = x[:, index:]

        return target_one_hot, target_features, non_sequence_features


class ReduceAndConcatTargetRBP(Layer):
    def __init__(self, num_rbp):
        super().__init__()
        self.use_rbp_scores = num_rbp > 0
        self.rpb_importance = tf.Variable(tf.zeros([1, num_rbp]))

    def call(self, target_one_hot, target_rbp_scores):

        # initialize output
        x = target_one_hot

        # if using RBP scores, check alignment and concatenate
        if self.use_rbp_scores:
            tf.assert_equal(tf.shape(target_one_hot)[1], tf.shape(target_rbp_scores)[1])
            w = tf.nn.softmax(self.rpb_importance)
            x = tf.concat([x, tf.reduce_sum(w * target_rbp_scores, axis=-1, keepdims=True)], axis=-1)

        return x


class TokenInputParser(Layer):
    def __init__(self, target_len: int, guide_len: int, use_guide_seq: bool):
        super().__init__()
        self.target_len = target_len
        self.guide_len = guide_len
        self.use_guide_seq = use_guide_seq

    def call(self, x):
        target_tokens = tf.cast(x[:, : self.target_len], tf.int32)
        i_parse = self.target_len
        target_positions = tf.cast(x[:, i_parse: i_parse + self.target_len], tf.int32)
        i_parse += self.target_len
        if self.use_guide_seq:
            guide_tokens = tf.cast(x[:, i_parse: i_parse + self.guide_len], tf.int32)
            i_parse += self.guide_len
        else:
            guide_tokens = None
        guide_positions = tf.cast(x[:, i_parse: i_parse + self.guide_len], tf.int32)
        i_parse += self.guide_len
        non_sequence_features = x[:, i_parse:]

        return target_tokens, target_positions, guide_tokens, guide_positions, non_sequence_features


class NucleotideAndPositionEncoding(Layer):
    def __init__(self, target_len: int, embedding_dim: int):
        super().__init__()

        # trainable embedding layers for nucleotide code and position
        self.nucleotide_embed = tf.keras.layers.Embedding(input_dim=4, output_dim=embedding_dim)
        self.position_embed = tf.keras.layers.Embedding(input_dim=target_len, output_dim=embedding_dim)

    def call(self, target_tokens, target_positions, guide_tokens, guide_positions, **kwargs):
        target_embedding = self.nucleotide_embed(target_tokens) + self.position_embed(target_positions)
        guide_embedding = self.position_embed(guide_positions)
        if guide_tokens is not None:
            guide_embedding += self.nucleotide_embed(guide_tokens)
        return target_embedding, guide_embedding


class TransformerEncoderLayer(Layer):
    def __init__(self, num_heads, dim_model, dim_hidden, dropout_rate=0.1):
        super().__init__()

        # layer components
        self.multi_head_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim_model)
        self.feed_fwd_net = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_hidden, activation='elu'),
            tf.keras.layers.Dense(dim_model)])
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, query, value, key, **kwargs):

        # multi-head attention output
        mha_output = self.multi_head_attn(query, value, key, **kwargs)
        mha_output = self.dropout_1(mha_output, **kwargs)
        mha_output = self.layer_norm_1(query + mha_output, **kwargs)

        # feed forward network output
        ffn_output = self.feed_fwd_net(mha_output, **kwargs)
        ffn_output = self.dropout_2(ffn_output, **kwargs)
        ffn_output = self.layer_norm_2(mha_output + ffn_output, **kwargs)

        return ffn_output


class TransformerDecoderLayer(TransformerEncoderLayer):
    def __init__(self, num_heads, dim_model, dim_hidden, dropout_rate=0.1):
        TransformerEncoderLayer.__init__(self, num_heads, dim_model, dim_hidden, dropout_rate)

        # additional layer components
        self.multi_head_self_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim_model)
        self.dropout_0 = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, query, value, key, **kwargs):

        # multi-head self attention output
        query = self.multi_head_self_attn(query, query, query, **kwargs)
        query = self.dropout_0(query, **kwargs)
        query = self.layer_norm_0(query + query, **kwargs)

        # run remaining components (identical to the encoder layer)
        out = TransformerEncoderLayer.call(self, query, value, key, **kwargs)

        return out


class TransformerLayer(Layer):
    def __init__(self, num_heads, dim_model, dim_hidden, num_layers, dropout_rate=0.1):
        super().__init__()

        self.enc_layers = []
        self.dec_layers = []
        for _ in range(num_layers):
            self.enc_layers += [TransformerEncoderLayer(num_heads, dim_model, dim_hidden, dropout_rate)]
            self.dec_layers += [TransformerDecoderLayer(num_heads, dim_model, dim_hidden, dropout_rate)]
        self.flatten = tf.keras.layers.Flatten()

    def call(self, encoder_in, decoder_in, **kwargs):
        for i in range(len(self.enc_layers)):
            encoder_in = self.enc_layers[i].call(query=encoder_in, key=encoder_in, value=encoder_in, **kwargs)
        for i in range(len(self.dec_layers) - 1):
            decoder_in = self.dec_layers[i].call(query=decoder_in, key=decoder_in, value=decoder_in, **kwargs)
        output = self.dec_layers[-1].call(query=encoder_in, key=decoder_in, value=decoder_in, **kwargs)

        return self.flatten(output)
