import os
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
from data import load_data, label_and_filter_data, model_inputs, SCALAR_FEATS
from models import build_model, train_model, test_model
from normalization import get_normalization_object

# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--gencode_batch_size', type=int, default=10000, help='gencode prediction batch size')
parser.add_argument('--sat_quant_active', type=float, default=0.1, help='set sigmoid(LFC) = q for quantile(LFC, q)')
parser.add_argument('--sat_quant_inactive', type=float, default=0.9, help='set sigmoid(LFC) = q for quantile(LFC, q)')
args = utils.parse_common_arguments(parser)
assert args.dataset == 'off-target'

# save path
output_dir = os.path.join('predictions', 'junction', 'tiger')
output_dir = os.path.join(output_dir, utils.data_directory(args.pm_only, args.indels, args.seq_only))
os.makedirs(output_dir, exist_ok=True)

# random seed
if args.seed is not None:
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)

# load and normalize training data
data_exon = load_data(dataset=args.dataset, pm_only=args.pm_only, indels=args.indels)
data_exon = label_and_filter_data(*data_exon, args.nt_quantile, args.filter_method, args.min_active_ratio)
normalizer = get_normalization_object(args.normalization)(df=data_exon, **args.normalization_kwargs)
data_exon = normalizer.normalize_targets(data_exon)

# load junction data
data_junc = load_data(dataset='junction', pm_only=args.pm_only, indels=args.indels, holdout=args.holdout)
data_junc = label_and_filter_data(*data_junc, args.nt_quantile, args.filter_method, args.min_active_ratio)

# load qpcr data
data_qpcr = pd.read_pickle(os.path.join('data-processed', 'junction-qpcr.bz2'))

# load gencode data
data_gencode, _ = load_data(dataset='junction-all', pm_only=args.pm_only, indels=args.indels)

# determine available scalar feature set
junction_features = set(list(data_junc.columns) + list(data_qpcr.columns) + list(data_gencode.columns))
if args.seq_only:
    available_features = set()
else:
    available_features = set(SCALAR_FEATS).intersection(set(data_exon.columns)).intersection(junction_features)
available_features = list(available_features)
available_features.sort()
print('Non-scalar features to be used:', available_features)
excluded_features = list((set(SCALAR_FEATS) - set(available_features)).intersection(set(data_junc.columns)))
excluded_features.sort()
print('Junction features excluded:', excluded_features)

# pre-validation
for i, fold in enumerate(set(data_exon['fold'].unique()) - {'training'}):

    # assemble model inputs
    train_data = model_inputs(data_exon[data_exon.fold != fold], args.context, scalar_feats=available_features)
    valid_data = model_inputs(data_exon[data_exon.fold == fold], args.context, scalar_feats=available_features)
    test_junc = data_junc.loc[~data_junc['guide_seq'].isin(data_exon.loc[data_exon.fold != fold, 'guide_seq'])]
    test_junc = model_inputs(test_junc, args.context, scalar_feats=available_features)
    test_qpcr = data_qpcr.loc[~data_qpcr['guide_seq'].isin(data_exon.loc[data_exon.fold != fold, 'guide_seq'])]
    test_qpcr = model_inputs(test_qpcr, args.context, scalar_feats=available_features)

    # build and train model
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
    test_model(model, valid_data).to_csv(os.path.join(output_dir, 'predictions_exon.csv'), **kwargs)
    test_model(model, test_junc).to_csv(os.path.join(output_dir, 'predictions_junc_ensemble.csv'), **kwargs)
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
pred_exon = pd.read_csv(os.path.join(output_dir, 'predictions_exon.csv'))
pred_junc = pd.read_csv(os.path.join(output_dir, 'predictions_junc_ensemble.csv'))
pred_qpcr = pd.read_csv(os.path.join(output_dir, 'predictions_qpcr_ensemble.csv'))
pred_gc = pd.read_csv(os.path.join(output_dir, 'predictions_gencode_ensemble.csv'))

# average ensembles
pred_junc = pred_junc.groupby(list(set(pred_junc.columns) - {'observed_lfc', 'predicted_lfc'})).mean().reset_index()
pred_qpcr = pred_qpcr.groupby(list(set(pred_qpcr.columns) - {'observed_lfc', 'predicted_lfc'})).mean().reset_index()
pred_gc = pred_gc.groupby(list(set(pred_gc.columns) - {'observed_lfc', 'predicted_lfc'})).mean().reset_index()

# fit sigmoid transformation to gencode predictions
lfc = pred_gc.loc[pred_gc['guide_type'] == 'PM', 'predicted_lfc']
x = np.array([[lfc.quantile(args.sat_quant_active), 1], [lfc.quantile(args.sat_quant_inactive), 1]])
y = np.log(np.array([[args.sat_quant_active], [args.sat_quant_inactive]]) ** -1 - 1)
a, b = np.squeeze(np.linalg.inv(x.T @ x) @ x.T @ y)
pred_gc['predicted_score'] = 1 - 1 / (1 + np.exp(a * pred_gc['predicted_lfc'] + b))

# score predictions and measure performance
for name, pred in [('exon', pred_exon), ('junc', pred_junc), ('qpcr', pred_qpcr)]:
    print('\n*** ' + name.upper() + ' ***')
    print('LFC: ', end='')
    utils.measure_performance(pred, pred_var='predicted_lfc')
    pred['predicted_score'] = 1 - 1 / (1 + np.exp(a * pred['predicted_lfc'] + b))
    print('Score: ', end='')
    utils.measure_performance(pred, pred_var='predicted_score')

# save results
pred_junc.to_csv(os.path.join(output_dir, 'predictions_junc.csv'), index=False)
pred_qpcr.to_csv(os.path.join(output_dir, 'predictions_qpcr.csv'), index=False)
pred_gc.to_csv(os.path.join(output_dir, 'predictions_gencode.csv'), index=False)
