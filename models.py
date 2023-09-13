import layers
import shap
import numpy as np
import pandas as pd
import tensorflow as tf
from data import NUCLEOTIDE_TOKENS, SCALAR_FEATS
from callbacks import PerformanceCallback
from typing import Union


# configure GPUs
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


class SequenceModelWithNonSequenceFeatures(object):
    def __init__(self):
        self.non_sequence_features = None

    def concatenate_non_sequence_features(self, data, x, scalar_feats):
        non_sequence_features = []
        for feature in (scalar_feats if self.non_sequence_features is None else self.non_sequence_features):
            if feature in data.keys():
                non_sequence_features.append(feature)
                x = tf.concat([x, tf.cast(data[feature][:, None], tf.float32)], axis=1)

        if self.non_sequence_features is None:
            self.non_sequence_features = non_sequence_features
        else:
            assert set(self.non_sequence_features) == set(non_sequence_features)

        return x


class OneHotSequenceModel(SequenceModelWithNonSequenceFeatures):
    def __init__(self, target_len: int, context_5p: int, context_3p: int, use_guide_seq: bool, pad_guide_seq: bool):
        super().__init__()
        self.input_parser = layers.OneHotInputParser(target_len, context_5p, context_3p, use_guide_seq, pad_guide_seq)

    def pack_inputs(self, data: dict, scalar_feats: Union[list, tuple] = tuple(SCALAR_FEATS)):

        # one-hot encode, flatten, and concatenate target (and context) sequence tokens
        x = tf.concat([data['5p_tokens'], data['target_tokens'], data['3p_tokens']], axis=1)
        x = tf.reshape(tf.one_hot(x, depth=4), [len(data['target_tokens']), -1])

        # if we are using guide sequence, do the same to those tokens (but pad them first to match target + context)
        if self.input_parser.guide_len > 0:

            # pre- and post-pad guides with zero to match combined target and context sequence
            data_type = data['guide_tokens'].dtype
            pad_5p = 255 * tf.ones([data['guide_tokens'].shape[0], data['5p_tokens'].shape[1]], dtype=data_type)
            pad_3p = 255 * tf.ones([data['guide_tokens'].shape[0], data['3p_tokens'].shape[1]], dtype=data_type)
            guide_tokens = tf.concat([pad_5p, data['guide_tokens'], pad_3p], axis=1)
            x = tf.concat([x, tf.reshape(tf.one_hot(guide_tokens, depth=4), [len(data['guide_tokens']), -1])], axis=1)

        # concatenate and log available non-sequence features
        x = self.concatenate_non_sequence_features(data, x, scalar_feats)

        # target values
        y = data['observed_lfc'] if 'observed_lfc' in data.keys() else None
        w = data['sample_weights'] if 'sample_weights' in data.keys() else None

        return x, y, w

    def parse_input_scores(self, scores):

        # unpack scores
        target_scores, guide_scores, non_sequence_scores = self.input_parser.call(scores)

        # load scores into DataFrame
        score_dict = dict()
        for nt, token in NUCLEOTIDE_TOKENS.items():
            score_dict.update({'target:' + nt: target_scores[..., token].numpy().tolist()})
            score_dict.update({'guide:' + nt: guide_scores[..., token].numpy().tolist()})
        for i, feature in enumerate(self.non_sequence_features):
            score_dict.update({feature: non_sequence_scores[:, i]})
        df = pd.DataFrame(score_dict)

        return df


class Tiger1D(OneHotSequenceModel):
    def __init__(self, target_len: int, context_5p: int, context_3p: int, use_guide_seq: bool, **kwargs):
        OneHotSequenceModel.__init__(self, target_len, context_5p, context_3p, use_guide_seq, pad_guide_seq=True)

        self.model = tf.keras.Sequential(name='Tiger1D', layers=[
            layers.SequenceSequentialWithNonSequenceBypass(
                input_parser=self.input_parser,
                sequence_layers=[
                    layers.AlignOneHotEncoding1D(use_guide_seq),
                    tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
                    tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
                    tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(0.25),
                ]),
            tf.keras.layers.Dense(units=128, activation='sigmoid'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=32, activation='sigmoid'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation=kwargs.get('output_fn') or 'linear')
        ])


class Tiger2D(OneHotSequenceModel):
    def __init__(self, target_len: int, context_5p: int, context_3p: int, use_guide_seq: bool, **kwargs):
        OneHotSequenceModel.__init__(self, target_len, context_5p, context_3p, use_guide_seq, pad_guide_seq=True)

        self.model = tf.keras.Sequential(name='Tiger2D', layers=[
            layers.SequenceSequentialWithNonSequenceBypass(
                input_parser=self.input_parser,
                sequence_layers=[
                    layers.AlignOneHotEncoding2D(use_guide_seq),
                    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu', padding='same'),
                    tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu', padding='same'),
                    tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding='same'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(0.25),
                ]),
            tf.keras.layers.Dense(units=128, activation='sigmoid'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=32, activation='sigmoid'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation=kwargs.get('output_fn') or 'linear')
        ])


class TargetSequenceWithRBP(SequenceModelWithNonSequenceFeatures):
    def __init__(self, guide_len: int, context_5p: int, context_3p: int, *, rbp_list: list, **kwargs):
        super().__init__()
        self.input_parser = layers.TargetSequenceAndPositionalFeatures(guide_len, context_5p, context_3p, len(rbp_list))
        self.rbp_list = rbp_list

        # model declaration
        self.model = tf.keras.Sequential(name='TargetSequenceWithRBP', layers=[
            layers.SequenceSequentialWithNonSequenceBypass(
                input_parser=self.input_parser,
                sequence_layers=[
                    layers.ReduceAndConcatTargetRBP(self.input_parser.feature_channels),
                    tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
                    tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
                    tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dropout(0.25),
                ]),
            tf.keras.layers.Dense(units=128, activation='sigmoid'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=32, activation='sigmoid'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='linear')
        ])

    def pack_inputs(self, data: dict, scalar_feats: Union[list, tuple] = tuple(SCALAR_FEATS)):

        # one-hot encode, flatten, and concatenate target (and context) sequence tokens
        x = tf.concat([data['5p_tokens'], data['target_tokens'], data['3p_tokens']], axis=1)
        x = tf.reshape(tf.one_hot(x, depth=4), [len(data['target_tokens']), -1])

        # if we are using additional non-sequence but positional target features
        if self.input_parser.feature_channels > 0:
            x = tf.concat([x, tf.reshape(data['target_features'], [len(data['target_tokens']), -1])], axis=1)

        # concatenate and log available non-sequence features
        x = self.concatenate_non_sequence_features(data, x, scalar_feats)

        # target values
        y = data['observed_lfc'] if 'observed_lfc' in data.keys() else None

        return x, y

    def parse_input_scores(self, scores):

        # unpack scores
        target_scores, target_feature_scores, non_sequence_scores = self.input_parser.call(scores)

        # load scores into DataFrame
        score_dict = dict()
        for nt, token in NUCLEOTIDE_TOKENS.items():
            score_dict.update({'target:' + nt: target_scores[..., token].numpy().tolist()})
        for i, rbp in enumerate(self.rbp_list):
            score_dict.update({rbp: target_feature_scores[..., i].numpy().tolist()})
        for i, feature in enumerate(self.non_sequence_features):
            score_dict.update({feature: non_sequence_scores[:, i]})
        df = pd.DataFrame(score_dict)

        return df


class TranscriptEmbeddingModel(SequenceModelWithNonSequenceFeatures):
    def __init__(self, target_len: int, guide_len: int, use_guide_seq: bool):
        super().__init__()
        self.input_parser = layers.TokenInputParser(target_len, guide_len, use_guide_seq)

    def pack_inputs(self, data: dict, scalar_feats: Union[list, tuple] = tuple(SCALAR_FEATS)):

        # concatenate target sequence and position tokens
        target_position = tf.range(start=0, limit=tf.shape(data['target_tokens'])[1], delta=1)
        target_position = tf.tile(target_position[None, :], [tf.shape(data['target_tokens'])[0], 1])
        x = tf.concat([tf.cast(data['target_tokens'], tf.float32), tf.cast(target_position, tf.float32)], axis=1)

        # concatenate guide sequence and position tokens
        if self.input_parser.use_guide_seq:
            x = tf.concat([x, tf.cast(data['guide_tokens'], tf.float32)], axis=1)
        guide_position = tf.range(start=0, limit=tf.shape(data['guide_tokens'])[1], delta=1)
        guide_position = tf.tile(guide_position[None, :], [tf.shape(data['guide_tokens'])[0], 1])
        x = tf.concat([x, tf.cast(guide_position, tf.float32)], axis=1)

        # concatenate and log available non-sequence features
        x = self.concatenate_non_sequence_features(data, x, scalar_feats)

        # target values
        y = data['observed_lfc']

        return x, y

    def parse_input_scores(self, x, scores):
        return pd.DataFrame()


class TranscriptTransformer(TranscriptEmbeddingModel):
    def __init__(self, target_len: int, guide_len: int, use_guide_seq: bool, **kwargs):
        TranscriptEmbeddingModel.__init__(self, target_len, guide_len, use_guide_seq)

        num_heads = kwargs.get('num_heads') or 10
        dim_model = kwargs.get('dim_model') or 16
        dim_hidden = kwargs.get('dim_hidden') or 16
        self.model = tf.keras.Sequential(name='TranscriptTransformer', layers=[
            layers.SequenceSequentialWithNonSequenceBypass(
                input_parser=self.input_parser,
                sequence_layers=[
                    layers.NucleotideAndPositionEncoding(target_len, embedding_dim=dim_model),
                    layers.TransformerLayer(num_heads, dim_model, dim_hidden, num_layers=3, dropout_rate=0.25),
                ]),
            tf.keras.layers.Dense(units=32, activation='elu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='linear')
        ])


def build_model(name, target_len, context_5p, context_3p, use_guide_seq, loss_fn, debug=False, **kwargs):
    if name == 'Tiger1D':
        model = Tiger1D(target_len, context_5p, context_3p, use_guide_seq, **kwargs)
        optimizer = tf.optimizers.Adam(1e-3)
    elif name == 'Tiger2D':
        model = Tiger2D(target_len, context_5p, context_3p, use_guide_seq, **kwargs)
        optimizer = tf.optimizers.Adam(1e-3)
    elif name == 'TargetSequenceWithRBP':
        model = TargetSequenceWithRBP(target_len, context_5p, context_3p, **kwargs)
        optimizer = tf.optimizers.Adam(1e-3)
    # elif name == 'TranscriptTransformer':
    #     model = TranscriptTransformer(target_len, guide_len, use_guide_seq, **kwargs)
    #     optimizer = tf.optimizers.Adam(5e-4)
    else:
        raise NotImplementedError
    model.model.compile(optimizer=optimizer, loss=loss_fn, weighted_metrics=[], run_eagerly=debug)

    return model


def train_model(model, train_data, valid_data, batch_size=2048, verbose=0, **kwargs):
    x, y, w = model.pack_inputs(train_data, **kwargs)
    model.model.fit(x, y, sample_weight=w, validation_data=model.pack_inputs(valid_data, **kwargs),
                    batch_size=batch_size, epochs=2000, verbose=verbose,
                    callbacks=[PerformanceCallback(same_line=not verbose, early_stop_patience=100)])
    return model


def test_model(model, valid_data):

    # keep relevant information
    df = pd.DataFrame()
    for key in ['gene', 'target_seq', 'guide_id', 'guide_seq', 'guide_type', 'observed_lfc', 'observed_label']:
        if key in valid_data.keys():
            df[key] = valid_data[key].numpy()
            if df[key].dtype == object:
                df[key] = df[key].apply(lambda s: s.decode('utf-8'))

    # generate predictions
    x, _, _ = model.pack_inputs(valid_data)
    df['predicted_lfc'] = model.model.predict(x, verbose=0)

    # if LFCs were predicted, join perfect match parents to each of their children (and themselves)
    if len(set(df['guide_type'].unique()) - {'PM'}) > 0 and 'predicted_lfc' in df.columns:
        df.set_index('target_seq', inplace=True)
        df_pm = df[df.guide_type == 'PM'].copy()
        assert not df_pm.index.has_duplicates  # make sure the indices (target sequences) are unique
        df_pm.rename(columns={'observed_lfc': 'observed_pm_lfc', 'predicted_lfc': 'predicted_pm_lfc'}, inplace=True)
        df_pm = df_pm[['observed_pm_lfc', 'predicted_pm_lfc']]
        df = df.join(df_pm, how='left')
        df.reset_index(inplace=True)

    return df


def explain_model(model, train_data, valid_data, num_background_samples=5000):

    # assemble inputs
    x_train, _, _ = model.pack_inputs(train_data)
    x_valid, _, _ = model.pack_inputs(valid_data)

    # select a set of background examples to take an expectation over
    num_background_samples = min(num_background_samples, x_train.shape[0])
    background = x_train.numpy()[np.random.choice(x_train.shape[0], num_background_samples, replace=False)]

    # compute Shapley values
    e = shap.DeepExplainer(model.model, background)
    shap_values = e.shap_values(x_valid.numpy())[0]

    # parse Shapley values into a DataFrame and append other relevant information
    df_shap = model.parse_input_scores(shap_values)
    current_cols = df_shap.columns.to_list()
    relevant_cols = ['gene', 'target_seq', 'guide_seq', 'guide_type']
    for col in relevant_cols:
        df_shap[col] = valid_data[col].numpy()
        if df_shap[col].dtype == object:
            df_shap[col] = df_shap[col].apply(lambda x: x.decode('utf-8'))
    df_shap = df_shap[relevant_cols + current_cols]

    return df_shap
