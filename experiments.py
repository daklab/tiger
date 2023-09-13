import os
import itertools
import utils
import zlib
import numpy as np
import pandas as pd
import tensorflow as tf
from data import load_data, label_and_filter_data, model_inputs, FEATURE_GROUPS, SCALAR_FEATS
from models import build_model, train_model, test_model, explain_model
from normalization import get_normalization_object

# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--experiment', type=str, default=None, help='which experiment to run')
parser.add_argument('--replace', action='store_true', default=False, help='forces rerun even if predictions exist')
parser.add_argument('--seed', type=int, default=12345, help='random number seed for reproducibility')
parser.add_argument('--seq_only', action='store_true', default=False, help='sequence only model')
args = utils.parse_common_arguments(parser)

# setup directories
data_sub_dir = utils.data_directory(args.pm_only, args.indels)
experiment_path = os.path.join('experiments', args.dataset, args.experiment, data_sub_dir, args.holdout)
os.makedirs(experiment_path, exist_ok=True)

# create or load data set with fold assignments
scale_non_seq_feats = args.experiment == 'SHAP'
data, data_nt = load_data(args.dataset, args.pm_only, args.indels, args.holdout, scale_non_seq_feats)
available_features = set(SCALAR_FEATS).intersection(set(data.columns))
if args.seq_only:
    available_features = set()

# define experimental values to test
if 'label-and-filter' in args.experiment:
    experimental_values = list(itertools.product(['MinActiveRatio'], [0.01, 0.05], np.arange(0.0, 0.35, 0.05)))
elif args.experiment == 'model':
    experimental_values = list(itertools.product(['Tiger1D', 'Tiger2D'], [set(), available_features]))
elif args.experiment == 'normalization':
    experimental_values = [
        ('No', dict()),
        ('FrequentistQuantile', dict(q_loc=50, q_neg=10, q_pos=90)),
        ('UnitInterval', dict(q_neg=0, q_pos=100, squash=False)),
        ('UnitInterval', dict(q_neg=1, q_pos=99, squash=False)),
        ('UnitInterval', dict(q_neg=5, q_pos=95, squash=False)),
        ('UnitInterval', dict(q_neg=10, q_pos=90, squash=False)),
        ('UnitInterval', dict(q_neg=1, q_pos=99, squash=True)),
        ('UnitInterval', dict(q_neg=5, q_pos=95, squash=True)),
        ('UnitInterval', dict(q_neg=10, q_pos=90, squash=True)),
        ('UnitVariance', dict()),
        ('ZeroMeanUnitVariance', dict()),
        # ('DepletionRatio', dict()),
        ('Sigmoid', dict(min_point=0.01, cutoff_point=0.95)),
        ('Sigmoid', dict(min_point=0.01, cutoff_point=0.90)),
        ('Sigmoid', dict(min_point=0.05, cutoff_point=0.95)),
        ('Sigmoid', dict(min_point=0.05, cutoff_point=0.90)),
        ('Sigmoid', dict(min_point=0.10, cutoff_point=0.95)),
        ('Sigmoid', dict(min_point=0.10, cutoff_point=0.90)),
    ]
elif args.experiment == 'context':
    experimental_values = [(-n, 0) if n < 0 else (0, n) for n in range(-25, 30, 5)]
    experimental_values += [(n, n) for n in range(5, 30, 5)]
    experimental_values += [(-n, 0) if n < 0 else (0, n) for n in [-4, -3, -2, -1, 1, 2, 3, 4]]
    experimental_values += [(n, n) for n in range(1, 5)]
    max_5p = max(data['5p_context'].apply(len))
    max_3p = max(data['3p_context'].apply(len))
    experimental_values = [(c5p, c3p) for c5p, c3p in experimental_values if c5p <= max_5p and c3p <= max_3p]
    experimental_values = list(itertools.product(experimental_values, [set(), available_features]))
elif args.experiment == 'feature-groups-individual':
    features = [tuple([feat for feat in available_features if feat in feats]) for feats in FEATURE_GROUPS.values()]
    experimental_values = [(g, f) for (g, f) in zip(FEATURE_GROUPS.keys(), features) if len(f) > 0]
    experimental_values = [('none', [])] + experimental_values
elif args.experiment == 'feature-groups-cumulative':
    individual_performance = os.path.join(experiment_path.replace('cumulative', 'individual'), 'performance.pkl')
    assert os.path.exists(individual_performance), 'run feature-groups-individual experiment first!'
    individual_performance = pd.read_pickle(individual_performance)
    groups = individual_performance['Pearson'].sort_values(ascending=False).index.get_level_values('feature group')
    groups = list(groups.drop('none'))
    experimental_values = [('none', [])]
    for group in groups:
        group_feats = [feat for feat in available_features if feat in FEATURE_GROUPS[group]]
        experimental_values += [(group, experimental_values[-1][1] + group_feats)]
elif args.experiment == 'learning-curve':
    experimental_values = np.arange(0.04, 1.04, 0.04)
elif args.experiment == 'SHAP':
    experimental_values = [None]
else:
    raise NotImplementedError

# initialize or load predictions
predictions_file = os.path.join(experiment_path, 'predictions.pkl')
df_predictions = pd.read_pickle(predictions_file) if os.path.exists(predictions_file) else pd.DataFrame()

# loop over experimental values
df_performance = pd.DataFrame()
df_shap = pd.DataFrame()
for experimental_value in experimental_values:
    print('**********', args.experiment, experimental_value, '**********')
    filter_method = experimental_value[0] if 'label-and-filter' in args.experiment else args.filter_method
    nt_quantile = experimental_value[1] if 'label-and-filter' in args.experiment else args.nt_quantile
    min_active_ratio = experimental_value[2] if 'label-and-filter' in args.experiment else args.min_active_ratio
    model_name = experimental_value[0] if args.experiment == 'model' else args.model
    normalization = experimental_value[0] if args.experiment == 'normalization' else args.normalization
    normalization_kwargs = experimental_value[1] if args.experiment == 'normalization' else args.normalization_kwargs
    context = experimental_value[0] if args.experiment == 'context' else args.context
    if args.experiment in {'context', 'model', 'feature-groups-individual', 'feature-groups-cumulative'}:
        features = [feature for feature in available_features if feature in experimental_value[1]]
    else:
        features = available_features
    training_utilization = experimental_value if args.experiment == 'learning-curve' else 1.0

    # set the configuration index
    config_dict = {
        'context': context,
        'features': tuple(np.sort(tuple(features))) if len(features) > 0 else 'None',
        'filter-method': filter_method,
        'loss': args.loss,
        'min active ratio': min_active_ratio,
        'model': model_name,
        'normalization': normalization,
        'normalization kwargs': str(normalization_kwargs),
        'non-targeting quantile': nt_quantile,
        'training utilization': training_utilization,
    }
    if args.experiment in {'feature-groups-individual', 'feature-groups-cumulative'}:
        config_dict.update({'feature group': experimental_value[0]})
    index = pd.MultiIndex.from_tuples([tuple(config_dict.values())], names=list(config_dict.keys()))

    # filter and normalize data
    filtered_data = label_and_filter_data(data, data_nt, nt_quantile, filter_method, min_active_ratio)
    normalizer = get_normalization_object(normalization)(filtered_data, **normalization_kwargs)
    normalized_data = normalizer.normalize(filtered_data)
    normalized_data = normalized_data.loc[~normalized_data.target_lfc.isna()]

    # add technical holdout
    if args.experiment == 'label-and-filter (off-target)':
        assert args.dataset == 'junction'
        assert args.normalization == 'No'
        normalized_data['fold'] = 'training'
        test_data = label_and_filter_data(*load_data('off-target', pm_only=True), nt_quantile=0.01, method='NoFilter')
        test_data = test_data.loc[~test_data['guide_seq'].isin(normalized_data['guide_seq'])]
        test_data['fold'] = 'test'
        normalized_data = pd.concat([normalized_data, test_data])

    # do we need results for this configuration?
    if args.replace or not index.isin(df_predictions.index.unique())[0]:

        # drop any existing results
        length_prior = len(df_predictions)
        df_predictions = df_predictions.loc[df_predictions.index.values != index.values]
        assert {length_prior - len(df_predictions)}.issubset({0, len(data)}), 'welp!'

        # loop over folds
        df_tap = pd.DataFrame()
        for k, fold in enumerate(set(normalized_data['fold'].unique()) - {'training'}):

            # a deterministic but seemingly random transformation of the experiment seed into a fold seed
            fold_seed = int(zlib.crc32(str(k * args.seed).encode())) % (2 ** 32 - 1)

            # prepare training and validation data
            tf.keras.utils.set_random_seed(fold_seed)
            train_data = normalized_data[normalized_data.fold != fold].sample(frac=training_utilization)
            train_data = model_inputs(train_data, context, scalar_feats=features)
            valid_data = model_inputs(normalized_data[normalized_data.fold == fold], context, scalar_feats=features)

            # train model
            tf.keras.utils.set_random_seed(fold_seed)
            model = build_model(name=model_name,
                                target_len=train_data['target_tokens'].shape[1],
                                context_5p=train_data['5p_tokens'].shape[1],
                                context_3p=train_data['3p_tokens'].shape[1],
                                use_guide_seq=args.use_guide_seq,
                                loss_fn=args.loss,
                                debug=args.debug,
                                output_fn=normalizer.output_fn)
            model = train_model(model, train_data, valid_data, args.batch_size)

            # accumulate targets and predictions on held-out fold
            df_tap = pd.concat([df_tap, test_model(model, valid_data)])

            # compute Shapley values if needed
            if args.experiment == 'SHAP':
                tf.keras.utils.set_random_seed(fold_seed)
                df_shap = pd.concat([df_shap, explain_model(model, train_data, valid_data)])

        # concatenate and save predictions
        df_tap.index = index.repeat(len(df_tap))
        df_predictions = pd.concat([df_predictions, normalizer.denormalize(df_tap.copy(deep=True))])
        df_predictions.to_pickle(predictions_file)

    # concatenate and save performance
    for normalized, active_only in itertools.product([False, True], [False, True]):
        df = df_predictions.loc[index].copy(deep=True)
        if normalized:
            df = normalizer.normalize(df)
        df = df.loc[df.target_label == 1] if active_only else df
        df = utils.measure_performance(df, index)
        df['Normalized'] = normalized
        df['Active Only'] = active_only
        df_performance = pd.concat([df_performance, df])
    df_performance.to_pickle(os.path.join(experiment_path, 'performance.pkl'))

# save Shapley values if needed
if args.experiment == 'SHAP':
    df_shap.to_pickle(os.path.join(experiment_path, 'shap.pkl'))
