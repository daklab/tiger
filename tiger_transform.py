import argparse
import os
import pickle
import hugging_face.tiger as tiger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from normalization import get_normalization_object
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tiger_figures import titration_confusion_matrix
from utils import string_to_dict, total_least_squares_slope, measure_performance, measure_guide_type_performance

# web-tool normalization settings
NORMALIZATION = 'UnitInterval'
NORMALIZATION_KWARGS = dict(q_neg=1, q_pos=99, squash=False)


def titration_ratio(df, correction=False, transpose=False):

    # total least squares slope correction
    if correction:
        x_name = 'predicted_lfc'
        y_name = 'target_lfc'
        for guide_type in {'PM', 'SM'}:
            x = df.loc[df.guide_type == guide_type, x_name].to_numpy()
            y = df.loc[df.guide_type == guide_type, y_name].to_numpy()
            slope = total_least_squares_slope(x, y)
            df.loc[df.guide_type == guide_type, x_name] = slope * df.loc[df.guide_type == guide_type, x_name]

    # keep PM and SM guides for high confidence essential genes only
    essential_genes = df.loc[df.guide_type == 'PM'].groupby('gene')['target_label'].sum()
    essential_genes = essential_genes.index.values[essential_genes > 6]
    df = df.loc[df.guide_type.isin({'PM', 'SM'}) & df.gene.isin(essential_genes)].copy()

    # only titer the top guides per transcript
    targets = []
    for gene in df['gene'].unique():
        index = (df.gene == gene) & (df.guide_type == 'PM')
        predictions = df.loc[index, 'predicted_lfc'].sort_values(ascending=not transpose).to_numpy()
        threshold = predictions[min(tiger.NUM_TOP_GUIDES - 1, len(predictions) - 1)]
        if transpose:
            targets += df.loc[index & (df.predicted_lfc >= threshold), 'target_seq'].to_list()
        else:
            targets += df.loc[index & (df.predicted_lfc <= threshold), 'target_seq'].to_list()
    df = df.loc[df.target_seq.isin(targets)].copy()

    # loop over target sites
    for target in df['target_seq'].unique():

        # target titration
        df.loc[df.target_seq == target, 'Observed ratio'] = 0
        index = (df.target_seq == target)
        df.loc[index, 'Observed ratio'] = df.loc[index, 'target_lfc'].rank(ascending=False, pct=True)
        z = df.loc[index & (df.guide_type == 'PM'), 'Observed ratio'].to_numpy()
        df.loc[index, 'Observed ratio'] /= z

        # predicted titration
        df.loc[df.target_seq == target, 'Predicted ratio'] = 0
        index = (df.target_seq == target)
        df.loc[index, 'Predicted ratio'] = df.loc[index, 'predicted_lfc'].rank(ascending=transpose, pct=True)
        z = df.loc[index & (df.guide_type == 'PM'), 'Predicted ratio'].to_numpy()
        df.loc[index, 'Predicted ratio'] /= z

    return df


def pick_best_normalization(use_current_best: bool):

    # load off-target predictions
    df_tap = pd.read_pickle('experiments/off-target/normalization/no_indels/targets/predictions.pkl')
    index = ['normalization', 'normalization kwargs']
    df_tap = df_tap.reset_index(index).set_index(index).sort_index()
    if use_current_best:
        df_tap = df_tap.loc[(NORMALIZATION, str(NORMALIZATION_KWARGS))].copy()

    # loop over normalization methods
    traces = pd.DataFrame()
    for index in df_tap.index.unique():

        # normalize predictions
        df = df_tap.loc[index].copy()
        kwargs = string_to_dict(index[1].replace('False', "'False'").replace('True', "'True'"))
        normalizer = get_normalization_object(index[0])(data=df, **kwargs)
        df_tap.loc[index] = normalizer.normalize(df)

        # compute titration accuracy
        for correction in [False, True]:
            df = titration_ratio(df_tap.loc[index].copy(), correction=correction)
            bins = np.arange(0.2, 1.0, .2)
            mtx = confusion_matrix(np.digitize(df['Observed ratio'], bins),
                                   np.digitize(df['Predicted ratio'], bins), normalize='pred')
            traces = pd.concat([traces, pd.DataFrame(dict(trace=np.trace(mtx), corrected=correction), index=[index])])

    # pick the model with the best titration performance
    best_index = traces.loc[~traces.corrected, 'trace'].idxmax()
    print('*** Normalization method with best titration performance ***')
    print(best_index)

    return df_tap.loc[best_index]


# def print_performance(df: pd.DataFrame, params: dict, title: str):
#     print('\n*** ' + title + ' ***')
#     print('*** Normalized predictions vs raw LFC performance ***')
#     measure_performance(df)
#     print('Active guides only: ', end='')
#     measure_performance(df[df.target_label == 1])
#     df['predicted_lfc'] = tiger.prediction_transform(df['predicted_lfc'].to_numpy(), **params)
#     print('*** Transformed predictions vs raw LFC performance ***')
#     measure_performance(df)
#     print('Active guides only: ', end='')
#     measure_performance(df[df.target_label == 1])


def calibrate_tiger(df_tap: pd.DataFrame):

    # loop over number of mismatches
    params = pd.DataFrame()
    for num_mismatches, codes in [(0, {'PM'}), (1, {'SM'}), (2, {'DM', 'RDM'}), (3, {'TM', 'RTM'})]:
        slope = measure_performance(df_tap.loc[df_tap.guide_type.isin(codes)], silence=True)['Slope']
        params = pd.concat([params, pd.DataFrame(dict(num_mismatches=[num_mismatches], slope=[float(slope)]))])

    # save parameters
    params.to_pickle(os.path.join('hugging_face', 'calibration_params.pkl'))

    return params


def test_calibration(df_tap: pd.DataFrame, params: pd.DataFrame, normalized_observations: bool = True):

    # performance before calibration
    print('\n*** Calibration Effect ***')
    print('Before:')
    measure_performance(df_tap)

    # apply calibration
    code_to_mismatches = dict(PM=0, SM=1, DM=2, RDM=2, TM=3, RTM=3)
    num_mismatches = df_tap['guide_type'].apply(lambda k: code_to_mismatches[k]).to_numpy()
    for col in ['predicted_lfc', 'predicted_pm_lfc']:
        df_tap[col] = tiger.calibrate_predictions(df_tap[col].to_numpy(), num_mismatches, params)

    # performance after calibration
    print('After:')
    measure_performance(df_tap)
    if normalized_observations:
        caption = 'Normalized Observed vs Calibrated Predicted'
    else:
        caption = 'Raw Observed vs Calibrated Predicted'
    titration_confusion_matrix(titration_ratio(df_tap.copy(), correction=False), caption)

    return df_tap


def transform_tiger(df_tap, active_saturation: float = 0.025, inactive_saturation: float = 0.975):

    # determine normalized activity saturation point
    lfc_active_pm = df_tap.loc[(df_tap.target_label == 1) & (df_tap.guide_type == 'PM'), 'predicted_lfc']
    active_cutoff = lfc_active_pm.quantile(active_saturation)

    # determine normalized inactivity saturation point
    lr = LogisticRegression(penalty=None, fit_intercept=True, class_weight='balanced')
    lr = lr.fit(df_tap[['predicted_lfc']], df_tap['target_label'])
    inactive_cutoff = float(-lr.intercept_ / lr.coef_)

    # save parameters according to method
    if tiger.UNIT_INTERVAL_MAP == 'sigmoid':
        x = np.array([[active_cutoff, 1], [inactive_cutoff, 1]])
        y = np.log(np.array([[active_saturation], [inactive_saturation]]) ** -1 - 1)
        a, b = np.squeeze(np.linalg.inv(x.T @ x) @ x.T @ y)
        params = dict(a=a, b=b)

    elif tiger.UNIT_INTERVAL_MAP == 'min-max':
        params = dict(a=df_tap['predicted_lfc'].min(), b=df_tap['predicted_lfc'].max())

    elif tiger.UNIT_INTERVAL_MAP == 'exp-lin-exp':
        params = dict(a=active_cutoff, b=active_saturation, c=inactive_cutoff, d=inactive_saturation)

    else:
        raise NotImplementedError

    # transform
    x = np.linspace(np.min(df_tap['predicted_lfc']), np.max(df_tap['predicted_lfc']), 1000)
    y = 1 - tiger.transform_predictions(x.copy(), params)

    # CDF all guides
    lfc_all = df_tap['predicted_lfc'].copy()
    lfc_all.to_numpy().sort()

    # CDF active guides
    lfc_active = df_tap.loc[df_tap.target_label == 1, 'predicted_lfc'].copy()
    lfc_active.to_numpy().sort()

    # CDF inactive guides
    lfc_inactive = df_tap.loc[df_tap.target_label == 0, 'predicted_lfc'].copy()
    lfc_inactive.to_numpy().sort()

    # CDF all PM guides
    lfc_pm_all = df_tap.loc[df_tap.guide_type == 'PM', 'predicted_lfc'].copy()
    lfc_pm_all.to_numpy().sort()

    # CDF active PM guides
    lfc_pm_active = df_tap.loc[(df_tap.guide_type == 'PM') & (df_tap.target_label == 1), 'predicted_lfc'].copy()
    lfc_pm_active.to_numpy().sort()

    # plot transform vs CDF
    plt.figure()
    plt.plot(x, y, label='1 - transform')
    plt.plot(lfc_all, np.cumsum(np.ones_like(lfc_all) / len(lfc_all)), label='CDF')
    plt.plot(lfc_active, np.cumsum(np.ones_like(lfc_active) / len(lfc_active)), label='CDF | active')
    plt.plot(lfc_inactive, np.cumsum(np.ones_like(lfc_inactive) / len(lfc_inactive)), label='CDF | inactive')
    plt.plot(lfc_pm_all, np.cumsum(np.ones_like(lfc_pm_all) / len(lfc_pm_all)), label='CDF | PM')
    plt.plot(lfc_pm_active, np.cumsum(np.ones_like(lfc_pm_active) / len(lfc_pm_active)), label='CDF | PM, active')
    plt.legend()

    # save parameters
    with open(os.path.join('hugging_face', 'transform_params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    return params


def test_transform(df_tap: pd.DataFrame, params: dict, normalized_observations: bool = True):

    # performance before/after transform
    print('\n*** Transformation Effect ***')
    print('Before:')
    measure_performance(df_tap)
    for col in ['predicted_lfc', 'predicted_pm_lfc']:
        df_tap[col] = tiger.transform_predictions(df_tap[col].to_numpy(), params)
    print('After:')
    measure_performance(df_tap)
    if normalized_observations:
        caption = 'Normalized Observed vs Transformed Predicted'
    else:
        caption = 'Raw Observed vs Transformed Predicted'
    titration_confusion_matrix(titration_ratio(df_tap.copy(), correction=False, transpose=True), caption)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_current_best', action='store_true', default=False)
    args = parser.parse_args()

    # pick normalization method with best calibration
    targets_and_predictions = pick_best_normalization(args.use_current_best)

    # plot titration w/ and w/o correction
    for correct, title in [(False, 'Normalized Observed vs Normalized Predicted'),
                           (True, 'Normalized Observed vs Corrected Predicted')]:
        titration_confusion_matrix(titration_ratio(targets_and_predictions.copy(), correction=correct), title)

    # calibrate tiger
    calibration_params = calibrate_tiger(targets_and_predictions.copy())
    targets_and_predictions = test_calibration(targets_and_predictions.copy(), calibration_params)

    # fit tiger transform
    transform_params = transform_tiger(targets_and_predictions.copy())
    test_transform(targets_and_predictions.copy(), transform_params)

    plt.show()
