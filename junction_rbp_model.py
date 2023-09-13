import os
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
from data import load_data, label_and_filter_data, model_inputs
from junction_utils import construct_junctions, append_rbp_predictions
from models import build_model, train_model, test_model, explain_model
from normalization import get_normalization_object

# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--mode', type=str, default='junction', help='operation mode')
parser.add_argument('--num_folds', type=int, default=10, help='number of validation folds')
parser.add_argument('--rbp_junc', action='store_true', default=False, help='use junction-level RBP predictions')
parser.add_argument('--rbp_nt', action='store_true', default=False, help='use nt-level RBP predictions')
parser.add_argument('--rbp_nt_relaxed', action='store_true', default=False, help='use relaxed nt-level RBP predictions')
parser.add_argument('--sum_peaks', action='store_true', default=False, help='one-hot encoded peaks along RBP axis')
args = utils.parse_common_arguments(parser)
context = (0, 0) if args.mode == 'junction' else (5, 5)
assert args.rbp_junc + args.rbp_nt + args.rbp_nt_relaxed <= 1

# random seed
if args.seed is not None:
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)

# setup directories
data_subdir = args.mode
if args.rbp_junc:
    data_subdir += '-rbp_junc'
elif args.rbp_nt:
    data_subdir += '-rbp_nt' + ('_sum_peaks' if args.sum_peaks else '')
elif args.rbp_nt_relaxed:
    data_subdir += '-rbp_nt_relaxed' + ('_sum_peaks' if args.sum_peaks else '')
else:
    data_subdir += '-seq_only'
experiment_path = os.path.join('experiments', 'junction', 'RBP', data_subdir)
os.makedirs(experiment_path, exist_ok=True)

# load and filter data
data = label_and_filter_data(*load_data('junction'), args.nt_quantile, args.filter_method, args.min_active_ratio)

# normalize data
normalizer = get_normalization_object(args.normalization)(data=data)
data = normalizer.normalize_targets(data)

# construct junction data
data = construct_junctions(data, reduce=(args.mode == 'junction'))

# load RBP data and concatenate RBP data if requested
scalar_feats = target_feats = list()
if args.rbp_junc:
    data, scalar_feats = append_rbp_predictions(data, suffix='junc')
elif args.rbp_nt:
    data, target_feats = append_rbp_predictions(data, 'nt', context[0], context[1], reduce=args.sum_peaks)
elif args.rbp_nt_relaxed:
    data, target_feats = append_rbp_predictions(data, 'nt_relaxed', context[0], context[1], reduce=args.sum_peaks)

# validation fold assignments
data['fold'] = 1 + np.random.choice(args.num_folds, size=len(data))

# loop over the folds
predictions = pd.DataFrame()
shap = pd.DataFrame()
for fold in range(1, args.num_folds + 1):

    # assemble model inputs
    train_data = model_inputs(data[data.fold != fold], context, scalar_feats=scalar_feats, target_feats=target_feats)
    valid_data = model_inputs(data[data.fold == fold], context, scalar_feats=scalar_feats, target_feats=target_feats)

    # build, train, and test model
    model = build_model(name='TargetSequenceWithRBP' if (args.rbp_nt or args.rbp_nt_relaxed) else 'Tiger1D',
                        target_len=train_data['target_tokens'].shape[1],
                        context_5p=train_data['5p_tokens'].shape[1],
                        context_3p=train_data['3p_tokens'].shape[1],
                        use_guide_seq=False,
                        loss_fn=args.loss,
                        rbp_list=target_feats,
                        debug=args.debug)  # TODO: output_fn
    model = train_model(model, train_data, valid_data, args.batch_size, scalar_feats=scalar_feats)
    df_tap = test_model(model, valid_data)
    predictions = pd.concat([predictions, normalizer.denormalize_targets_and_predictions(df_tap)])

    # SHAP values
    shap = pd.concat([shap, explain_model(model, train_data, valid_data, num_background_samples=5000)])


# measure performance
print('Performance:')
performance = utils.measure_performance(predictions)

# save
predictions.to_pickle(os.path.join(experiment_path, 'predictions.pkl'))
performance.to_pickle(os.path.join(experiment_path, 'performance.pkl'))
shap.to_pickle(os.path.join(experiment_path, 'shap.pkl'))
