import os
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
from data import load_data, label_and_filter_data, model_inputs, FEATURE_GROUPS, SCALAR_FEATS
from models import build_model, train_model, test_model
from normalization import get_normalization_object

# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--gencode_batch_size', type=int, default=10000, help='gencode prediction batch size')
parser.add_argument('--sat_quant_active', type=float, default=0.1, help='set sigmoid(LFC) = q for quantile(LFC, q)')
parser.add_argument('--sat_quant_inactive', type=float, default=0.9, help='set sigmoid(LFC) = q for quantile(LFC, q)')
parser.add_argument('--use_lfc', action='store_true', default=False, help='Use Day 21 LFC instead of sea-bass LMM slope')
args = utils.parse_common_arguments(parser)
assert args.dataset == 'junction'
assert args.pm_only

# save path
output_dir = os.path.join('predictions', 'junction', 'tiger-junc' if args.use_lfc else 'tiger-bass')
output_dir = os.path.join(output_dir, utils.data_directory(args.pm_only, args.indels, args.seq_only))
os.makedirs(output_dir, exist_ok=True)

# random seed
if args.seed is not None:
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)

# load training data and merge sea-bass slope estimates
data_junc = load_data(args.dataset, pm_only=args.pm_only, indels=args.indels, holdout=args.holdout)
data_junc = label_and_filter_data(*data_junc, args.nt_quantile, args.filter_method, args.min_active_ratio)
sea_bass = pd.read_pickle(os.path.join('data-processed', 'sea-bass.bz2'))
sea_bass_cols = list(set(sea_bass.columns) - set(data_junc.columns))
data_junc = data_junc.merge(sea_bass[['gene', 'guide_seq'] + sea_bass_cols], how='inner', on=['gene', 'guide_seq'])
if not args.use_lfc:
    data_junc['observed_lfc'] = data_junc['slope']
normalizer = get_normalization_object(args.normalization)(df=data_junc, **args.normalization_kwargs)
data_junc = normalizer.normalize_targets(data_junc)
data_junc = data_junc.loc[~np.isnan(data_junc['observed_lfc']) & ~np.isinf(data_junc['observed_lfc'])]

# load qpcr data
data_qpcr = pd.read_pickle(os.path.join('data-processed', 'junction-qpcr.bz2'))

# load gencode data
data_gencode, _ = load_data(dataset='junction-all', pm_only=args.pm_only, indels=args.indels)

# determine available scalar feature set
junction_features = set(list(data_junc.columns) + list(data_qpcr.columns) + list(data_gencode.columns))
if args.seq_only:
    available_features = set()
else:
    available_features = set(SCALAR_FEATS).intersection(junction_features) - set(FEATURE_GROUPS['junction overlap'])
available_features = list(available_features)
available_features.sort()
print('Non-scalar features to be used:', available_features)
excluded_features = list((set(SCALAR_FEATS) - set(available_features)).intersection(set(data_junc.columns)))
excluded_features.sort()
print('Junction features excluded:', excluded_features)

# pre-validation
for i, fold in enumerate(set(data_junc['fold'].unique()) - {'training'}):

    # assemble model inputs
    train_data = model_inputs(data_junc[data_junc.fold != fold], args.context, scalar_feats=available_features)
    valid_data = model_inputs(data_junc[data_junc.fold == fold], args.context, scalar_feats=available_features)
    test_qpcr = data_qpcr.loc[~data_qpcr['guide_seq'].isin(data_junc.loc[data_junc.fold != fold, 'guide_seq'])]
    test_qpcr = model_inputs(test_qpcr, args.context, scalar_feats=available_features)

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

    # accumulate predictions
    kwargs = dict(header=(i == 0), index=False, mode='w' if (i == 0) else 'a')
    test_model(model, valid_data).to_csv(os.path.join(output_dir, 'predictions_junc_cv.csv'), **kwargs)
    test_model(model, test_qpcr).to_csv(os.path.join(output_dir, 'predictions_qpcr_ensemble.csv'), **kwargs)
    for j in range(0, len(data_gencode), args.gencode_batch_size):
        j_stop = min(j + args.gencode_batch_size, len(data_gencode))
        df = data_gencode.iloc[j:j_stop]
        df = test_model(model, model_inputs(df, args.context, scalar_feats=available_features))
        df.to_csv(os.path.join(output_dir, 'predictions_gencode_ensemble.csv'), **kwargs)
        print('\rGencode percent complete: {:.2f}%'.format(100 * j_stop / len(data_gencode)), end='')

    # free keras memory
    tf.keras.backend.clear_session()

# load predictions
pred_junc = pd.read_csv(os.path.join(output_dir, 'predictions_junc_cv.csv'))
pred_qpcr = pd.read_csv(os.path.join(output_dir, 'predictions_qpcr_ensemble.csv'))
pred_gc = pd.read_csv(os.path.join(output_dir, 'predictions_gencode_ensemble.csv'))

# average ensembles
pred_qpcr = pred_qpcr.groupby(list(set(pred_qpcr.columns) - {'observed_lfc', 'predicted_lfc'})).mean().reset_index()
pred_gc = pred_gc.groupby(list(set(pred_gc.columns) - {'observed_lfc', 'predicted_lfc'})).mean().reset_index()

# fit sigmoid transformation to gencode predictions
lfc = pred_gc.loc[pred_gc['guide_type'] == 'PM', 'predicted_lfc']
x = np.array([[lfc.quantile(args.sat_quant_active), 1], [lfc.quantile(args.sat_quant_inactive), 1]])
y = np.log(np.array([[args.sat_quant_active], [args.sat_quant_inactive]]) ** -1 - 1)
a, b = np.squeeze(np.linalg.inv(x.T @ x) @ x.T @ y)
pred_gc['predicted_score'] = 1 - 1 / (1 + np.exp(a * pred_gc['predicted_lfc'] + b))

# score predictions and measure performance
for name, pred in [('junc', pred_junc), ('qpcr', pred_qpcr)]:
    print('\n*** ' + name.upper() + ' ***')
    print('LFC: ', end='')
    utils.measure_performance(pred, pred_var='predicted_lfc')
    pred['predicted_score'] = 1 - 1 / (1 + np.exp(a * pred['predicted_lfc'] + b))
    print('Score: ', end='')
    utils.measure_performance(pred, pred_var='predicted_score')

# denormalize junction observations and targets
pred_junc = normalizer.denormalize_targets_and_predictions(pred_junc)

# save results
pred_junc.to_csv(os.path.join(output_dir, 'predictions_junc.csv'), index=False)
pred_qpcr.to_csv(os.path.join(output_dir, 'predictions_qpcr.csv'), index=False)
pred_gc.to_csv(os.path.join(output_dir, 'predictions_gencode.csv'), index=False)
