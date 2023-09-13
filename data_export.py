import pickle
import utils
import pandas as pd
import numpy as np
import tensorflow as tf
from data import load_data, label_and_filter_data, model_inputs, NUCLEOTIDE_TOKENS, SCALAR_FEATS
from normalization import get_normalization_object

# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--junction_mode', type=str, default='targets', help='how to treat junction data')
args = utils.parse_common_arguments(parser)

# load data set
data, data_nt = load_data(args.dataset, pm_only=args.pm_only, indels=args.indels)
available_features = set(SCALAR_FEATS).intersection(set(data.columns))
data = label_and_filter_data(data, data_nt, args.nt_quantile, args.filter_method, args.min_active_ratio)

# for junction datasets, perform required additional preprocessing
if 'junction' in args.dataset:
    if args.junction_mode == 'splice-sites':
        data['target_seq'] = data['5p_context'] + data['target_seq'] + data['3p_context']
        data_junc = data[['target_seq', 'target_lfc', 'target_label']].groupby('target_seq').mean()
        data_junc['target_label'] = data_junc['target_label'] >= 0.25
        del data['target_lfc'], data['target_label']
        data = pd.merge(data_junc, data, on='target_seq')
        data = data.set_index('target_seq')
        data = data[~data.index.duplicated(keep='first')].reset_index()
        target_context = (0, 0)
    elif args.junction_mode == 'targets':
        target_context = (15, 0)
    else:
        raise NotImplementedError
    args.dataset = args.dataset + '-' + args.junction_mode
else:
    target_context = (15, 0)

# normalize data
normalizer = get_normalization_object(args.normalization)(data=data)
data = normalizer.normalize(data)

# assemble model inputs
data = model_inputs(data, target_context, scalar_feats=set(), include_replicates=True)
data['target_tokens'] = tf.concat([data['5p_tokens'], data['target_tokens'], data['3p_tokens']], axis=1)

# utilized sequence as a string for motif analysis
token_to_nt = np.array(['N'] * 256)
for nt, token in NUCLEOTIDE_TOKENS.items():
    token_to_nt[token] = nt
sequence = tf.gather(tf.constant(token_to_nt, tf.string), tf.cast(data['target_tokens'], tf.int32))
sequence = tf.strings.reduce_join(sequence, axis=-1)

# save redacted version
data = dict(x=data['target_tokens'],
            y_mean=tf.expand_dims(data['target_lfc'], 1),
            y_replicates=data['replicate_lfc'],
            sequence=sequence)
with open(args.dataset + '.pkl', 'wb') as f:
    pickle.dump(data, f)
