import os
import utils
import pandas as pd
import tensorflow as tf
from data import load_data, label_and_filter_data, model_inputs, SCALAR_FEATS
from models import build_model, train_model, test_model
from normalization import get_normalization_object

# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--mm_only', action='store_true', default=False, help='use only mismatched guides')
args = utils.parse_common_arguments(parser)
assert not (args.mm_only and args.pm_only)

# random seed
if args.seed is not None:
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)

# setup output directory
sub_dir = 'mm' if args.mm_only else utils.data_directory(args.pm_only, args.indels, args.seq_only)
pred_path = os.path.join('predictions', args.dataset, sub_dir, args.holdout)
os.makedirs(pred_path, exist_ok=True)

# load, label, and filter data
data = load_data(args.dataset, pm_only=args.pm_only, indels=args.indels, holdout=args.holdout)
data = label_and_filter_data(*data, args.nt_quantile, args.filter_method, args.min_active_ratio)

# normalize data
normalizer = get_normalization_object(args.normalization)(data, **args.normalization_kwargs)
data = normalizer.normalize_targets(data)

# remove PM guides if requested
if args.mm_only:
    data = data[data.guide_type != 'PM']

# available features
available_features = set(SCALAR_FEATS).intersection(set(data.columns))

# pre-validation of dataset
predictions = pd.DataFrame()
for fold in set(data['fold'].unique()) - {'training'}:

    # prepare training and validation data
    train_data = model_inputs(data[data.fold != fold], args.context, scalar_feats=available_features)
    valid_data = model_inputs(data[data.fold == fold], args.context, scalar_feats=available_features)

    # train model
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

    # accumulate targets and predictions for held-out fold
    predictions = pd.concat([predictions, test_model(model, valid_data)])

    # free keras memory
    tf.keras.backend.clear_session()

# undo gene essentiality normalization
predictions = normalizer.denormalize_targets_and_predictions(predictions)

# save predictions and performance
predictions.to_pickle(os.path.join(pred_path, 'predictions.pkl'))
predictions.to_csv(os.path.join(pred_path, 'predictions.csv'), index=False)
performance = utils.measure_performance(predictions)
performance.to_pickle(os.path.join(pred_path, 'performance.pkl'))
