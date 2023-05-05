import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import norm
from statsmodels.stats.diagnostic import lilliefors
from tensorflow.keras.preprocessing.sequence import pad_sequences

# project-wide nucleotide tokens
NUCLEOTIDE_TOKENS = dict(zip(['A', 'C', 'G', 'T'], [0, 1, 2, 3]))

# relevant columns
INDEX_COLS = ['gene', 'guide_id']
LFC_COLS = ['lfc_r1', 'lfc_r2', 'lfc_r3']
SEQUENCE_FEATS = ['guide_seq', 'target_seq', '5p_context', '3p_context']
FEATURE_GROUPS = {
    'target location': [
        'loc_utr_5p',
        'loc_cds',
        'loc_utr_3p',
        'log_gene_len'],
    'junction proximity': [
        'junction_dist_5p',
        'junction_dist_3p'],
    'guide secondary structure': [
        'direct_repeat',
        'g_quad'],
    'guide MFE': [
        'mfe'],
    'hybridization MFE': [
        'hybrid_mfe_1_23',
        'hybrid_mfe_15_9',
        'hybrid_mfe_3_12'],
    'target accessibility': [
        'log_unpaired',
        'log_unpaired_11',
        'log_unpaired_19',
        'log_unpaired_25'],
}
SCALAR_FEATS = list()
for features in FEATURE_GROUPS.values():
    SCALAR_FEATS += features
SCALAR_FEATS_INDELS = set(SCALAR_FEATS) - {'hybrid_mfe_1_23', 'hybrid_mfe_15_9', 'hybrid_mfe_3_12'}
UNIT_SCALED_FEATS = ['loc_utr_5p', 'loc_cds', 'loc_utr_3p',
                     'direct_repeat', 'g_quad',
                     'perc_gene_nuc', 'perc_junc_nuc', 'perc_junc_use']


def sequence_complement(sequence: str) -> str:
    nt_complement = dict(zip(['A', 'C', 'G', 'T'], ['T', 'G', 'C', 'A']))
    return ''.join([nt_complement[nt] for nt in sequence])


def load_data(dataset, pm_only=False, indels=False, holdout='targets', scale_non_seq_feats=False):
    """
    Loads specified dataset and its corresponding non-targeting data (if it exists)
    :param dataset: which dataset to use
    :param pm_only: only include perfectly matched guides
    :param indels: whether to include guides with insertions or deletions
    :param holdout: what should be held out in each fold {genes, guides, targets}
    :param scale_non_seq_feats: whether to scale non-sequence features to unit interval
    :return: two DataFrames containing targeting and non-targeting data, respectively
    """
    data_file = os.path.join('data-processed', dataset + '.bz2')
    assert os.path.exists(data_file), 'if assertion fails, run: python data.py'

    # load data
    data = pd.read_pickle(data_file)

    # filtered data as requested
    if pm_only:
        data = data[data.guide_type == 'PM']
    elif not indels:
        data = data[data.guide_type.isin(['PM', 'SM', 'DM', 'RDM', 'TM', 'RTM'])]

    # set the folds
    if holdout == 'genes':
        data['fold'] = data['gene']
    elif holdout == 'guides':
        data['fold'] = data['guide_fold']
    elif holdout == 'targets':
        data['fold'] = data['target_fold']
    else:
        raise NotImplementedError

    # load non-targeting data if it exists
    nt_file = os.path.join('data-processed', dataset + '-nt.bz2')
    data_nt = pd.read_pickle(nt_file) if os.path.exists(nt_file) else None

    # scalar feature transformation and scaling
    for feature in SCALAR_FEATS:
        if feature in data.columns:
            if 'junction_dist' in feature:
                data[feature] = data[feature].apply(lambda x: np.log10(1 + np.abs(x)))
            if scale_non_seq_feats and feature not in UNIT_SCALED_FEATS:
                data[feature] -= data[feature].min()
                scale = data[feature].max()
                if scale > 0:
                    data[feature] /= scale
                print(feature, data[feature].min(), data[feature].max())

    return data, data_nt


def label_and_filter_data(data, data_nt, nt_quantile=0.01, method='MinActiveRatio', min_active_ratio=0.1, quiet=True):
    """
    Labels guide activity as active (LFC < specified non-targeting quantile) and filters our non-essential genes
    :param data: screen data as a DataFrame
    :param data_nt: non-targeting data as a DataFrame
    :param nt_quantile: non-targeting quantile that defines the threshold under which guides are considered active
    :param method: essential gene filter method
    :param min_active_ratio: used by MinActiveRatio--genes with an active guide ratio less than this value get removed
    :param quiet: silences Lilliefors non-targeting distribution tests
    :return: filtered data with target labels (active vs inactive)
    """
    # non-targeting data is available
    if data_nt is not None and len(data_nt) > 0:

        # compute mean of replicates
        data_nt['lfc'] = data_nt[LFC_COLS].mean(axis=1)

        # set active threshold based on quantile of non-targeting distribution (assumed to be normal)
        threshold = norm.ppf(q=nt_quantile, loc=data_nt['lfc'].mean(), scale=data_nt['lfc'].std())
        if not quiet:
            _, p_val = lilliefors(data_nt['lfc'].values)
            print('Lilliefors p-value of NT replicate medians: {:.4e}'.format(p_val))
            print('A {:.4f} quantile yields an LFC threshold of {:.4f}'.format(nt_quantile, threshold))

    # non-targeting data is unavailable, so use default threshold
    else:
        threshold = -0.5

    # provided target values
    if set(LFC_COLS).issubset(data.columns):

        # take mean of replicates as target value
        data['target_lfc'] = data[LFC_COLS].mean(axis=1)
        data = data[~data['target_lfc'].isna()]
        assert sum(np.isnan(data['target_lfc'].values)) == 0

        # label guides as active/inactive
        data['target_label'] = data['target_lfc'] < threshold

        # # label mismatch titration
        # data_pm = data[data['guide_type'] == 'PM'][['gene', 'target_seq', 'target_lfc']]
        # data_pm.rename(columns={'target_lfc': 'pm_lfc'}, inplace=True)
        # data = pd.merge(data, data_pm, on=['gene', 'target_seq'])
        # data['target_titration'] = 2 ** (data['pm_lfc'] - data['target_lfc'])

    # apply filter
    if method == 'NoFilter':
        return data
    elif method == 'MinActiveRatio' and 'target_label' in data.columns:
        df = pd.DataFrame(data.groupby('gene')['target_label'].mean().rename('active ratio'))
        return data[data['gene'].isin(df[df['active ratio'] >= min_active_ratio].index.values)]
    else:
        raise NotImplementedError


def model_inputs(data, target_context, scalar_features, include_replicates=False, max_context=100):
    """
    Prepares a dictionary of model inputs and target values from the provided DataFrame
    :param data: panda's DataFrame containing model inputs and target values
    :param target_context: amount of target context
    :param scalar_features: scalar features to be provided to the model
    :param include_replicates: whether to include raw replicates
    :param max_context: maximum amount of up- and down-stream context (reduce RAM usage for guides with 1kb available)
    :return: dictionary containing model inputs and target values
    """
    # shuffle rows (tensorflow shuffling is approximate, doing a full shuffle here makes it more exact)
    data = data.sample(frac=1)

    # keep only data without NaN values for the scalar features
    data = data[(~data[list(scalar_features)].isna()).product(1) == 1]

    # trim context to some reasonable amount
    data['5p_context'] = data['5p_context'].apply(lambda s: s[-min(len(s), max_context):]).astype(str)
    data['3p_context'] = data['3p_context'].apply(lambda s: s[:min(len(s), max_context)]).astype(str)

    # load target and guide sequences
    target_seq = tf.stack([tf.constant(list(seq)) for seq in data['target_seq']], axis=0)
    left_context = tf.ragged.stack([tf.constant(list(seq), tf.string) for seq in data['5p_context']], axis=0)
    right_context = tf.ragged.stack([tf.constant(list(seq), tf.string) for seq in data['3p_context']], axis=0)
    guide_seq = tf.ragged.stack([tf.constant(list(seq)) for seq in data['guide_seq']], axis=0)

    # convert nucleotides to integer codes
    nucleotide_table = tf.lookup.StaticVocabularyTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(NUCLEOTIDE_TOKENS.keys()) + ['N'], dtype=tf.string),
            values=tf.constant(list(NUCLEOTIDE_TOKENS.values()) + [255], dtype=tf.int64)),
        num_oov_buckets=1)
    target_tokens = nucleotide_table.lookup(target_seq)
    tokens_5p = nucleotide_table.lookup(left_context)
    tokens_3p = nucleotide_table.lookup(right_context)
    guide_tokens = tf.RaggedTensor.from_row_splits(values=nucleotide_table.lookup(guide_seq.values),
                                                   row_splits=guide_seq.row_splits).to_tensor(255)
    target_tokens = tf.cast(target_tokens, tf.uint8)
    guide_tokens = tf.cast(guide_tokens, tf.uint8)

    # these operations are only necessary if using additional target context sequence
    if isinstance(target_context, (list, tuple)):
        context_5p, context_3p = tuple(target_context)
    else:
        context_5p = context_3p = target_context

    # add target context
    tokens_5p = tf.constant(pad_sequences(tokens_5p.to_list(), dtype='uint8', padding='pre', value=255))
    tokens_5p = tokens_5p[:, tokens_5p.shape[1]-context_5p:tokens_5p.shape[1]]
    tokens_3p = tf.constant(pad_sequences(tokens_3p.to_list(), dtype='uint8', padding='post', value=255))
    tokens_3p = tokens_3p[:, :context_3p]

    # assemble dictionary of core model inputs
    inputs = {
        # data identifiers for downstream analysis
        'gene': tf.constant(data['gene'], tf.string),
        'target_seq': tf.constant(data['target_seq'], tf.string),
        'guide_seq': tf.constant(data['guide_seq'], tf.string),
        'guide_type': tf.constant(data['guide_type'], tf.string),
        # sequence features
        'target_tokens': target_tokens,
        'guide_tokens': guide_tokens,
        '5p_tokens': tokens_5p,
        '3p_tokens': tokens_3p,
    }

    # target values
    if 'target_lfc' in data.columns:
        inputs.update({'target_lfc': tf.constant(data['target_lfc'], tf.float32)})
    if 'target_label' in data.columns:
        inputs.update({'target_label': tf.constant(data['target_label'], tf.uint8)})
    if include_replicates:
        inputs.update({'replicate_lfc': tf.constant(data[LFC_COLS], tf.float32)})

    # add optional features
    for feature in scalar_features:
        if feature in data.columns:
            inputs.update({feature: tf.constant(data[feature], tf.float32)})
        else:
            raise Exception('Missing feature: ' + feature)

    return inputs
