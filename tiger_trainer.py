import os
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import hugging_face.tiger as hf
from data import load_data, label_and_filter_data, training_validation_split_targets, model_inputs
from matplotlib import pyplot as plt
from models import build_model, train_model, test_model
from normalization import get_normalization_object
from sklearn.linear_model import LogisticRegression
from tiger_figures import titration_confusion_matrix
from utils import measure_performance, titration_ratio

# web-tool normalization settings
NORMALIZATION = 'UnitInterval'
NORMALIZATION_KWARGS = dict(q_neg=1, q_pos=99, squash=False)


def calibrate_tiger(df: pd.DataFrame):

    # loop over number of mismatches and compute slope between predictions and observations
    params = pd.DataFrame()
    for num_mismatches, codes in [(0, {'PM'}), (1, {'SM'}), (2, {'DM', 'RDM'}), (3, {'TM', 'RTM'})]:
        slope = measure_performance(df.loc[df.guide_type.isin(codes)], silence=True)['Slope']
        params = pd.concat([params, pd.DataFrame(dict(num_mismatches=[num_mismatches], slope=[float(slope)]))])

    return params


def test_calibration(df: pd.DataFrame, params: pd.DataFrame, title: str = ''):

    # performance before calibration
    print('\n*** Calibration Effect ***')
    print('Before:')
    measure_performance(df)

    # apply calibration
    code_to_mismatches = dict(PM=0, SM=1, DM=2, RDM=2, TM=3, RTM=3)
    num_mismatches = df['guide_type'].apply(lambda k: code_to_mismatches[k]).to_numpy()
    for col in ['predicted_lfc', 'predicted_pm_lfc']:
        if col in df.columns:
            df[col] = hf.calibrate_predictions(df[col].to_numpy(), num_mismatches, params)

    # performance after calibration
    print('After:')
    measure_performance(df)
    titration_confusion_matrix(titration_ratio(df.copy(), num_top_guides=hf.NUM_TOP_GUIDES), title)


def score_tiger(df: pd.DataFrame, sat_quant_active: float, sat_quant_inactive: float):
    assert 0 < sat_quant_active < sat_quant_inactive < 1

    # map upper and lower predicted LFC quantiles to their respective locations on a sigmoid
    lfc = df.loc[df['guide_type'] == 'PM', 'predicted_lfc']
    x = np.array([[lfc.quantile(sat_quant_active), 1], [lfc.quantile(sat_quant_inactive), 1]])
    y = np.log(np.array([[sat_quant_active], [sat_quant_inactive]]) ** -1 - 1)
    a, b = np.squeeze(np.linalg.inv(x.T @ x) @ x.T @ y)

    return pd.DataFrame(dict(a=[a], b=[b]))


def decision_boundary(df: pd.DataFrame):
    # determine active threshold
    lr = LogisticRegression(penalty=None, fit_intercept=True, class_weight='balanced')
    lr = lr.fit(df[['predicted_lfc']], df['observed_label'])
    return float(-lr.intercept_ / lr.coef_)


def test_scoring(df: pd.DataFrame, params: pd.DataFrame, title: str = ''):

    # performance before/after transform
    print('\n*** Transformation Effect ***')
    print('Before:')
    print('Active threshold: {:.4f}'.format(decision_boundary(df)))
    measure_performance(df)
    for col in ['predicted_lfc', 'predicted_pm_lfc']:
        if col in df.columns:
            df[col] = hf.score_predictions(df[col].to_numpy(), params)
    print('After:')
    print('Active threshold: {:.4f}'.format(decision_boundary(df)))
    measure_performance(df)
    titration_confusion_matrix(titration_ratio(df.copy(), num_top_guides=hf.NUM_TOP_GUIDES, transpose=True), title)


if __name__ == '__main__':

    # script arguments
    parser = utils.common_parser_arguments()
    parser.add_argument('--sat_quant_active', type=float, default=0.05, help='sigmoid(LFC) := q for quantile(LFC, q)')
    parser.add_argument('--sat_quant_inactive', type=float, default=0.95, help='sigmoid(LFC) := q for quantile(LFC, q)')
    parser.add_argument('--random_seed', type=int, default=12345, help='random number seed')
    parser.add_argument('--retrain', action='store_true', default=False, help='retrain TIGER')
    parser.add_argument('--training_ratio', type=float, default=0.9, help='ratio of data for training')
    args = utils.parse_common_arguments(parser)
    assert 0 < args.training_ratio < 1

    # random seed
    if args.random_seed is not None:
        tf.config.experimental.enable_op_determinism()
        tf.keras.utils.set_random_seed(args.random_seed)

    # load, label, filter, and split data
    data = load_data(dataset='off-target', pm_only=False, indels=False)
    data = label_and_filter_data(*data, method='NoFilter')
    data = training_validation_split_targets(data, train_ratio=0.9)

    # normalize data
    normalizer = get_normalization_object(NORMALIZATION)(data, **NORMALIZATION_KWARGS)
    data = normalizer.normalize_targets(data)

    # assemble model inputs
    context = (hf.CONTEXT_5P, hf.CONTEXT_3P)
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

    # (re)train and save model
    model_path = os.path.join('hugging_face', 'model')
    if args.retrain or not os.path.exists(model_path):
        tiger = train_model(tiger, train_data, valid_data, args.batch_size)
        tiger.model.save(model_path, overwrite=True, include_optimizer=False, save_format='tf')

    # or load existing model
    else:
        tiger.model.load_weights(model_path)

    # generate predictions
    df_tap = test_model(tiger, valid_data)

    # normalized titration performance
    titration_confusion_matrix(titration_ratio(df_tap.copy(), num_top_guides=hf.NUM_TOP_GUIDES),
                               title='Normalized HEK293 Observations vs Normalized Predictions')

    # mismatch specific calibration
    calibration_params = calibrate_tiger(df_tap.copy())
    if args.retrain:
        calibration_params.to_pickle(os.path.join('hugging_face', 'calibration_params.pkl'))
    test_calibration(df_tap, calibration_params, title='Normalized HEK293 Observations vs Calibrated Predictions')

    # map predicted LFC to the unit interval (i.e. score predictions)
    scoring_params = score_tiger(df_tap.copy(), args.sat_quant_active, args.sat_quant_inactive)
    if args.retrain:
        scoring_params.to_pickle(os.path.join('hugging_face', 'scoring_params.pkl'))
    test_scoring(df_tap, scoring_params, title='Normalized HEK293 Observations vs Scored Predictions')

    # HAP1 test set
    test_data = load_data(dataset='hap-titration', pm_only=False, indels=False)
    test_data = label_and_filter_data(*test_data, method='MinActiveRatio', min_active_ratio=1.0)
    test_data = model_inputs(test_data, context=context, scalar_feats=set())
    df_tap = test_model(tiger, test_data)
    titration_confusion_matrix(titration_ratio(df_tap.copy(), num_top_guides=hf.NUM_TOP_GUIDES),
                               title='HAP1 Titration vs Normalized Predictions')
    test_calibration(df_tap, calibration_params, title='HAP1 Titration vs Calibrated Predictions')
    test_scoring(df_tap, scoring_params, title='HAP1 Titration vs Scored Predictions')

    plt.show()
