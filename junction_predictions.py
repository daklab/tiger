import os
import gc
import utils
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from data import load_data, label_and_filter_data, model_inputs
from models import build_model, train_model, test_model
from normalization import get_normalization_object
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--training_set', type=str, default='off-target', help='which dataset to use')
parser.add_argument('--correct', action='store_true', default=False, help='corrects for proliferation differences')
parser.add_argument('--num_folds', type=int, default=10, help='number of validation folds')
parser.add_argument('--num_trials', type=int, default=1, help='number of trials')
parser.add_argument('--replace', action='store_true', default=False, help='forces rerun even if predictions exist')
parser.add_argument('--seed', type=int, default=None, help='random number seed for reproducibility')
args = utils.parse_common_arguments(parser, dataset_parameter_override='junction')
args.pm_only = True

# output files
save_dir = os.path.join('predictions', 'junction')
os.makedirs(save_dir, exist_ok=True)
prefix = args.training_set + ('-corrected' if args.correct else '')
predictions_csv_gencode = os.path.join(save_dir, prefix + '-predictions-gencode.csv')
predictions_csv_junction = os.path.join(save_dir, prefix + '-predictions-junction.csv')
predictions_csv_off_target = os.path.join(save_dir, prefix + '-predictions-off-target.csv')
for file in [predictions_csv_gencode, predictions_csv_junction, predictions_csv_off_target]:
    if os.path.exists(file):
        if args.replace:
            os.remove(file)
        else:
            raise FileExistsError
performance_pkl = os.path.join(save_dir, prefix + '-performance.pkl')

# random seed
if args.seed is not None:
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.seed)

# load and normalize junction data
data_junction = load_data(dataset='junction', pm_only=args.pm_only, indels=args.indels)
data_junction = label_and_filter_data(*data_junction,
                                      nt_quantile=args.nt_quantile,
                                      method=args.filter_method,
                                      min_active_ratio=args.min_active_ratio)
normalizer_junction = get_normalization_object(args.normalization)(data=data_junction)
data_junction = normalizer_junction.normalize(data_junction)

# load and normalize off-target data
data_off_target = load_data(dataset='off-target', pm_only=args.pm_only, indels=args.indels)
data_off_target = label_and_filter_data(*data_off_target, nt_quantile=args.nt_quantile, method='NoFilter')
normalizer_off_target = get_normalization_object(args.normalization)(data=data_off_target)
data_off_target = normalizer_off_target.normalize(data_off_target)

# if requested, correct for cell proliferation differences
if args.correct:

    # correction parameterization
    def func(x, a, b):
        return a * x + (1 - np.exp(x)) * b

    # plot the correction
    data_junction['cell'] = 'A375'
    data_off_target['cell'] = 'HEK293'
    data_common = pd.concat([data_off_target, data_junction])
    data_common = data_common.pivot(index=['gene', 'guide_seq'], columns=['cell'], values='target_lfc')
    g = sns.jointplot(data=data_common, x='HEK293', y='A375', joint_kws=dict(alpha=0.5))
    data_common.dropna(inplace=True)
    popt = curve_fit(func, xdata=data_common['HEK293'], ydata=data_common['A375'])[0]
    x = np.linspace(data_common['HEK293'].min(), data_common['HEK293'].max())
    g.ax_joint.plot(x, func(x, *popt), label='Correction')
    g.ax_joint.plot(x, x, color='black', linestyle=':')
    g.figure.suptitle('Proliferation Correction')
    plt.tight_layout()
    g.figure.savefig(os.path.join(save_dir, 'proliferation_correction.pdf'))
    plt.show()

    # apply the correction
    data_off_target['target_lfc'] = data_off_target['target_lfc'].apply(lambda lfc: func(lfc, *popt))

# load test data
data_gencode = pd.read_pickle(os.path.join('data-processed', 'junction-all.bz2'))
data_gencode['fold'] = 1 + np.random.choice(args.num_folds, size=len(data_gencode))

# set up indices
data_gencode = data_gencode.reset_index().set_index('guide_id')
data_junction = data_junction.reset_index().set_index('guide_id')
data_off_target = data_off_target.reset_index().set_index('guide_id')  # it doesn't matter if this has duplicates
assert not data_gencode.index.has_duplicates
assert not data_junction.index.has_duplicates

# loop over the trials
performance = pd.DataFrame()
for trial in range(1, args.num_trials + 1):

    # correct folds for dataset overlaps
    index_junction = data_junction['guide_seq'].isin(data_off_target['guide_seq'])
    index_off_target = data_off_target['guide_seq'].isin(data_junction['guide_seq'])
    data_off_target.loc[index_off_target, 'fold'] = data_junction.loc[index_junction, 'fold']
    data_gencode.loc[data_junction.index, 'fold'] = data_junction['fold']

    # loop over the folds
    predictions_junction = pd.DataFrame()
    predictions_off_target = pd.DataFrame()
    for fold in range(1, args.num_folds + 1):
        gc.collect()
        tf.keras.backend.clear_session()
        print('***** Trial {:}/{:} | Fold {:}/{:} *****'.format(trial, args.num_trials, fold, args.num_folds))

        # construct training and validation inputs
        if args.training_set == 'combined':
            data_combined = pd.concat([data_junction, data_off_target])
            train_inputs = data_combined.loc[data_combined.fold != fold]
            valid_inputs = data_combined.loc[data_combined.fold == fold]
            del data_combined
        elif args.training_set == 'junction':
            train_inputs = data_junction.loc[data_junction.fold != fold]
            valid_inputs = data_junction.loc[data_junction.fold == fold]
        elif args.training_set == 'off-target':
            train_inputs = data_off_target.loc[data_off_target.fold != fold]
            valid_inputs = data_off_target.loc[data_off_target.fold == fold]
        else:
            raise NotImplementedError
        train_inputs = model_inputs(train_inputs.reset_index(), args.context, scalar_feats=set())
        valid_inputs = model_inputs(valid_inputs.reset_index(), args.context, scalar_feats=set())

        # build and train model
        model = build_model(name=args.model,
                            target_len=train_inputs['target_tokens'].shape[1],
                            context_5p=train_inputs['5p_tokens'].shape[1],
                            context_3p=train_inputs['3p_tokens'].shape[1],
                            use_guide_seq=args.use_guide_seq,
                            loss_fn=args.loss,
                            debug=args.debug,
                            output_fn=normalizer.output_fn,
                            **args.kwargs) # TODO: output_fn
        model = train_model(model, train_inputs, valid_inputs, args.batch_size)
        del train_inputs

        # accumulate predictions for validation folds
        datasets = [data_gencode, data_junction, data_off_target]
        csv_files = [predictions_csv_gencode, predictions_csv_junction, predictions_csv_off_target]
        for dataset, csv_file in zip(datasets, csv_files):
            valid_inputs = model_inputs(dataset[dataset.fold == fold].reset_index(), args.context, scalar_feats=set())
            predictions = test_model(model, valid_inputs)
            del valid_inputs
            predictions['trial'] = trial
            predictions.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file))
            del predictions

    # measure trial's pre-validation performance
    for dataset, csv_file in zip(['junction', 'off-target'], [predictions_csv_junction, predictions_csv_off_target]):
        predictions = pd.read_csv(csv_file, index_col=0)
        predictions = predictions.loc[predictions.trial == trial]
        index = pd.MultiIndex.from_frame(pd.DataFrame(dict(dataset=[dataset], trial=[trial])))
        performance = pd.concat([performance, utils.measure_performance(predictions, index=index)])

# save performance
performance.to_pickle(performance_pkl)
