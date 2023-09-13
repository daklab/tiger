import os
import pickle
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
from data import label_and_filter_data, load_data, model_inputs
from matplotlib import pyplot as plt
from models import build_model, train_model, test_model
from normalization import get_normalization_object
from hugging_face.tiger import CONTEXT_5P, CONTEXT_3P
from tiger_figures import titration_confusion_matrix
from tiger_transform import NORMALIZATION, NORMALIZATION_KWARGS, titration_ratio, test_calibration, test_transform


# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--random_seed', type=int, default=12345, help='random number seed')
parser.add_argument('--training_ratio', type=float, default=0.9, help='ratio of data for training')
args = utils.parse_common_arguments(parser)
assert 0 < args.training_ratio < 1

# random seed
if args.random_seed is not None:
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.random_seed)

# load, label, and filter data
data = load_data(dataset='off-target', pm_only=False, indels=False)
data = label_and_filter_data(*data, method='NoFilter')
del data['fold']

# target site folds
targets = data[data.guide_type == 'PM'][['target_seq']].reset_index()
targets['fold'] = np.random.choice(['training', 'validation'], p=[0.9, 0.1], size=len(targets))
data = pd.merge(data, targets, how='inner', on='target_seq')

# normalize data
normalizer = get_normalization_object(NORMALIZATION)(data, **NORMALIZATION_KWARGS)
data = normalizer.normalize(data)

# assemble model inputs
context = (CONTEXT_5P, CONTEXT_3P)
train_data = model_inputs(data[data.fold == 'training'], context=context, scalar_feats=set())
valid_data = model_inputs(data[data.fold == 'validation'], context=context, scalar_feats=set())

# build model
tiger = build_model(name='Tiger2D',
                    target_len=train_data['target_tokens'].shape[1],
                    context_5p=train_data['5p_tokens'].shape[1],
                    context_3p=train_data['3p_tokens'].shape[1],
                    use_guide_seq=True,
                    loss_fn='log_cosh',
                    debug=args.debug,
                    output_fn=normalizer.output_fn,
                    **args.kwargs)

# train and save
tiger = train_model(tiger, train_data, valid_data, args.batch_size)
tiger.model.save(os.path.join('hugging_face', 'model'), overwrite=True, include_optimizer=False, save_format='tf')

# generate predictions
df_tap = test_model(tiger, valid_data)

# normalized titration performance
title = 'Normalized Observations vs Normalized Predictions'
titration_confusion_matrix(titration_ratio(df_tap.copy()), title)

# denormalize observed data
df_tap = normalizer.denormalize(df_tap, cols=['target_lfc', 'target_pm_lfc'])

# titration ratio without normalized observations
title = 'Raw Observations vs Normalized Predictions'
titration_confusion_matrix(titration_ratio(df_tap.copy()), title)

# calibration effect
calibration_params = pd.read_pickle(os.path.join('hugging_face', 'calibration_params.pkl'))
test_calibration(df_tap, calibration_params, normalized_observations=False)

# transform effect
with open(os.path.join('hugging_face', 'transform_params.pkl'), 'rb') as f:
    transform_params = pickle.load(f)
test_transform(df_tap, transform_params, normalized_observations=False)

plt.show()
