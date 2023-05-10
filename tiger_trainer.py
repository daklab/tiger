import os
import utils
import numpy as np
import tensorflow as tf
from data import label_and_filter_data, load_data, model_inputs
from models import build_model, train_model, test_model
from normalization import get_normalization_object
from tiger.tiger import CONTEXT_5P, CONTEXT_3P

# script arguments
parser = utils.common_parser_arguments()
parser.add_argument('--random_seed', type=int, default=12345, help='random number seed')
args = utils.parse_common_arguments(parser)

# random seed
if args.random_seed is not None:
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(args.random_seed)

# load, label, and filter data
data = load_data(dataset='off-target', pm_only=False, indels=False)
data = label_and_filter_data(*data, method='NoFilter')

# 90-10 split validation split
data['fold'] = np.random.choice(np.array(['training', 'validation']), p=[0.9, 0.1], size=len(data))

# normalize data
normalizer = get_normalization_object('UnitVariance')(data=data)
data = normalizer.normalize(data)

# assemble model inputs
context = (CONTEXT_5P, CONTEXT_3P)
train_data = model_inputs(data[data.fold == 'training'], target_context=context, scalar_features=set())
valid_data = model_inputs(data[data.fold == 'validation'], target_context=context, scalar_features=set())

# build model
tiger = build_model(name='Tiger2D',
                    target_len=train_data['target_tokens'].shape[1],
                    context_5p=train_data['5p_tokens'].shape[1],
                    context_3p=train_data['3p_tokens'].shape[1],
                    use_guide_seq=True,
                    loss_fn='log_cosh',
                    debug=args.debug,
                    **args.kwargs)

# train and save
tiger = train_model(tiger, train_data, valid_data, args.batch_size)
tiger.model.save(os.path.join('hugging-face', 'model'), overwrite=True, include_optimizer=False, save_format='tf')

# measure performance
df_tap = test_model(tiger, valid_data)
print('Normalized: ', end='')
utils.measure_performance(df_tap)
print('Original: ', end='')
utils.measure_performance(normalizer.denormalize(df_tap))
