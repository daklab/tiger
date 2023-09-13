import os
import utils
import pandas as pd
import tensorflow as tf
from data import load_data, label_and_filter_data, model_inputs
from models import build_model, train_model, test_model, explain_model
from utils import common_parser_arguments, parse_common_arguments, measure_performance
from normalization import get_normalization_object

# script arguments
parser = common_parser_arguments()
parser.add_argument('--abridged', action='store_true', default=False, help='abridged junction sequence')
parser.add_argument('--random_seed', type=int, default=None, help='random number seed')
parser.add_argument('--replace', action='store_true', default=False, help='forces rerun even if results exist')
args = parse_common_arguments(parser)

# random seed
if args.random_seed is not None:
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.random_seed)

# setup directories
data_sub_dir = utils.data_directory(args.pm_only, args.indels)
experiment_dir = 'tiger-team-abridged' if args.abridged else 'tiger-team'
experiment_path = os.path.join('experiments', 'junction', experiment_dir, data_sub_dir, args.holdout)
os.makedirs(experiment_path, exist_ok=True)

# load and filter data
data = label_and_filter_data(*load_data('junction'), args.nt_quantile, args.filter_method, args.min_active_ratio)
assert data['guide_seq'].nunique() == len(data), 'pd.merge calls rely on this fact below'

# normalize data
normalizer = get_normalization_object(args.normalization)(data=data)
data = normalizer.normalize(data)

# add junction sequences and their target values
data['junction_seq'] = data['5p_context'] + data['target_seq'] + data['3p_context']
if args.abridged:
    data['junction_seq'] = data['junction_seq'].apply(lambda s: s[20:70])
data_junc = data[['junction_seq', 'target_lfc', 'target_label']].groupby('junction_seq').mean()
data_junc['target_label'] = data_junc['target_label'] >= 0.25
data_junc.rename(columns={'target_lfc': 'junction_lfc', 'target_label': 'junction_label'}, inplace=True)
data = data.set_index('junction_seq').join(data_junc).reset_index()

# if we can load existing results, do so
results_exist = not args.replace
results_exist &= os.path.exists(os.path.join(experiment_path, 'predictions.pkl'))
results_exist &= os.path.exists(os.path.join(experiment_path, 'shap.pkl'))
if results_exist:
    predictions = pd.read_pickle(os.path.join(experiment_path, 'predictions.pkl'))
    shap = pd.read_pickle(os.path.join(experiment_path, 'shap.pkl'))

# otherwise, generate results
else:

    # loop over the tasks
    predictions = pd.DataFrame()
    shap = pd.DataFrame()
    for task in ['guide mean', 'junction mean', 'guide residual', 'junction residual']:
        index = pd.Index(data=[task], name='task')

        # prepare task specific data
        data_task = data.copy(deep=True)
        if task == 'junction mean':
            data_task['target_lfc'] = data_task['junction_lfc']
            data_task['target_label'] = data_task['junction_label']
            data_task['target_seq'] = data_task['junction_seq']
            data_task.drop_duplicates('target_seq', inplace=True)
        elif task == 'guide residual':
            junction_means = predictions.loc['junction mean', ['target_seq', 'predicted_lfc']]
            junction_means.rename(columns={'target_seq': 'junction_seq'}, inplace=True)
            data_task = pd.merge(data_task, junction_means, how='left', on='junction_seq')
            data_task['target_lfc'] = data_task['target_lfc'] - data_task['predicted_lfc']
        elif task == 'junction residual':
            guide_means = predictions.loc['guide mean', ['target_seq', 'predicted_lfc']]
            data_task = pd.merge(data_task, guide_means, how='left', on='target_seq')
            data_task['target_lfc'] = data_task['target_lfc'] - data_task['predicted_lfc']
            df = data_task[['junction_seq', 'target_lfc']].groupby('junction_seq').mean()
            data_task = data_task.join(df, 'junction_seq', lsuffix='_ignore')
            data_task['target_seq'] = data_task['junction_seq']
            data_task.drop_duplicates('target_seq', inplace=True)

        # loop over folds
        for fold in set(data['fold'].unique()) - {'training'}:
            print(task + ' | fold: ' + str(fold))

            # split data and build pipeline
            train_data = model_inputs(data_task[data_task['fold'] != fold], context=0, scalar_feats=set())
            valid_data = model_inputs(data_task[data_task['fold'] == fold], context=0, scalar_feats=set())

            # build, train, and test model
            model = build_model(name=args.model,
                                target_len=train_data['target_tokens'].shape[1],
                                context_5p=train_data['5p_tokens'].shape[1],
                                context_3p=train_data['3p_tokens'].shape[1],
                                use_guide_seq=False,
                                loss_fn=args.loss,
                                debug=args.debug)
            model = train_model(model, train_data, valid_data, args.batch_size)
            predictions_fold = test_model(model, valid_data)
            shap_fold = explain_model(model, train_data, valid_data, num_background_samples=10000)

            # accumulate predictions
            predictions = pd.concat([predictions, predictions_fold.set_index(index.repeat(len(predictions_fold)))])
            shap = pd.concat([shap, shap_fold.set_index(index.repeat(len(shap_fold)))])

    # save results
    predictions.to_pickle(os.path.join(experiment_path, 'predictions.pkl'))
    shap.to_pickle(os.path.join(experiment_path, 'shap.pkl'))

# compute results
performance = pd.DataFrame()
for task in predictions.index.unique():

    # measure performance (don't denormalize residual prediction performance)
    if task in {'guide mean', 'junction mean'}:
        results = normalizer.denormalize(predictions.loc[task])
        results_index = pd.Index(data=[task], name='task')
        performance = pd.concat([performance, measure_performance(results, results_index, silence=True)])
    else:
        df = predictions.loc[task]
        del df['target_label']
        results_index = pd.Index(data=[task], name='task')
        performance = pd.concat([performance, measure_performance(df, results_index, silence=True)])

    # covert guide predictions to junction predictions (and vice-a-versa)
    if task == 'guide mean':
        df_tap_junc = predictions.loc[task, ['gene', 'target_seq', 'predicted_lfc']]
        df_tap_junc = data.set_index(['gene', 'target_seq']).join(df_tap_junc.set_index(['gene', 'target_seq']))
        df_tap_junc = df_tap_junc.reset_index().groupby(['gene', 'junction_seq']).mean().reset_index()
        df_tap_junc['target_label'] = df_tap_junc['target_label'] >= 0.25
        df_tap_junc = normalizer.denormalize(df_tap_junc)
        results_index = pd.Index(data=['guide mean predicting junction mean'], name='task')
        performance = pd.concat([performance, measure_performance(df_tap_junc, results_index, silence=True)])

    elif task == 'junction mean':
        df_tap_guide = predictions.loc[task, ['target_seq', 'predicted_lfc']]
        df_tap_guide.rename(columns={'target_seq': 'junction_seq'}, inplace=True)
        df_tap_guide = data.set_index('junction_seq').join(df_tap_guide.set_index('junction_seq'))
        df_tap_guide = normalizer.denormalize(df_tap_guide)
        results_index = pd.Index(data=['junction mean predicting guide mean'], name='task')
        performance = pd.concat([performance, measure_performance(df_tap_guide, results_index, silence=True)])

# measure combined performance
for guide_task, junction_task in [('residual', 'mean'), ('mean', 'residual')]:
    df_tap_guide = predictions.loc['guide ' + guide_task, ['guide_seq', 'predicted_lfc']]
    df_tap_guide = pd.merge(df_tap_guide, data, how='left', on='guide_seq')
    df_tap_junc = predictions.loc['junction ' + junction_task, ['target_seq', 'predicted_lfc']]
    df_tap_junc.rename(columns={'target_seq': 'junction_seq'}, inplace=True)
    df_tap_combined = pd.merge(df_tap_guide, df_tap_junc, how='left', on='junction_seq', suffixes=('_guide', '_junc'))
    df_tap_combined['predicted_lfc'] = df_tap_combined['predicted_lfc_guide'] + df_tap_combined['predicted_lfc_junc']
    df_tap_combined = normalizer.denormalize(df_tap_combined)
    task = ('guide mean + junction residual' if guide_task == 'mean' else 'junction mean + guide residual')
    results_index = pd.Index(data=[task], name='task')
    performance = pd.concat([performance, measure_performance(df_tap_combined, results_index, silence=True)])

# print/save performance
print(performance[['Pearson', 'Spearman', 'AUROC', 'AUPRC']])
performance.to_pickle(os.path.join(experiment_path, 'performance.pkl'))
