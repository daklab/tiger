import os
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
from data import load_data, label_and_filter_data, model_inputs, LFC_COLS, SCALAR_FEATS
from models import build_model, train_model, test_model
from normalization import get_normalization_object

# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--mm_only', action='store_true', default=False, help='use only mismatched guides')
parser.add_argument('--test_dataset', type=str, default=None, help='optional held out test set')
parser.add_argument('--test_filter_method', type=str, default='NoFilter', help='gene filtering method for test set')
args = utils.parse_common_arguments(parser)

# random seed
if args.seed is not None:
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)

# load, label, filter, and split data
data = load_data(dataset=args.dataset, pm_only=args.pm_only, indels=args.indels)
data = label_and_filter_data(*data, args.nt_quantile, args.filter_method, args.min_active_ratio)
available_features = set(SCALAR_FEATS).intersection(set(data.columns))
data['fold'] = np.random.choice(np.array(['training', 'validation']), p=[0.9, 0.1], size=len(data))

# normalize data
normalizer = get_normalization_object(args.normalization)(data, **args.normalization_kwargs)
data = normalizer.normalize_targets(data)

# set up test set similarly, if requested
if args.test_dataset:
    test_data = load_data(dataset=args.test_dataset, pm_only=args.pm_only, indels=args.indels)
    test_data = label_and_filter_data(*test_data, args.nt_quantile, args.test_filter_method)
    available_features = available_features.intersection(set(test_data.columns))
    if set(LFC_COLS).issubset(test_data.columns):
        test_set_normalizer = get_normalization_object(args.normalization)(test_data, **args.normalization_kwargs)
        test_data = test_set_normalizer.normalize_targets(test_data)
    else:
        test_set_normalizer = None
else:
    test_data = None
    test_set_normalizer = None

# remove PM guides if requested
if args.mm_only:
    data = data[data.guide_type != 'PM']
    test_data = None or test_data[test_data.guide_type != 'PM']

# print which non-scalar features will be used
available_features = set() if args.seq_only else available_features
print('Non-scalar features to be used:', available_features)

# assemble model inputs
train_data = model_inputs(data[data.fold == 'training'], args.context, scalar_feats=available_features)
valid_data = model_inputs(data[data.fold == 'validation'], args.context, scalar_feats=available_features)

# build, train, and test model
model = build_model(name=args.model,
                    target_len=train_data['target_tokens'].shape[1],
                    context_5p=train_data['5p_tokens'].shape[1],
                    context_3p=train_data['3p_tokens'].shape[1],
                    use_guide_seq=args.use_guide_seq,
                    loss_fn=args.loss,
                    debug=args.debug,
                    output_fn=normalizer.output_fn,
                    **args.kwargs)
model = train_model(model, train_data, valid_data, args.batch_size)
df_tap = test_model(model, valid_data)

# undo gene normalization
df_tap = normalizer.denormalize_targets_and_predictions(df_tap)

# measure performance
print('Validation performance:')
utils.measure_performance(df_tap)

# if a test dataset was specified
if test_data is not None:

    # setup output directory
    sub_dir = 'mm' if args.mm_only else utils.data_directory(args.pm_only, args.indels, args.seq_only)
    pred_path = os.path.join('predictions', args.dataset, sub_dir, args.test_dataset)
    os.makedirs(pred_path, exist_ok=True)

    # get normalized predictions
    normalized_columns = ['observed_lfc', 'observed_pm_lfc', 'predicted_lfc', 'predicted_pm_lfc']
    df_tap = test_model(model, model_inputs(test_data, args.context, scalar_feats=available_features))

    # if we have a test set normalizer
    if test_set_normalizer is not None:
        df_tap = test_set_normalizer.denormalize_targets_and_predictions(df_tap)
        print('Test performance:')
        df_performance = utils.measure_performance(df_tap)
        df_performance.to_pickle(os.path.join(pred_path, 'performance.pkl'))
    else:
        df_tap = df_tap.rename(columns=dict(zip(normalized_columns, [s + '_normalized' for s in normalized_columns])))

    # save predictions
    df_tap.to_pickle(os.path.join(pred_path, 'predictions.pkl'))
    df_tap.to_csv(os.path.join(pred_path, 'predictions.csv'), index=False)
