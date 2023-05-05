import argparse
import json
import numpy as np
import pandas as pd
from data import load_data
from scipy.stats import ks_2samp, norm, pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_curve, roc_auc_score
from roc_comparison.compare_auc_delong_xu import delong_roc_test, delong_roc_variance


def common_parser_arguments():

    # common arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='tensorflow batch size')
    parser.add_argument('--context', type=str, default='(1,1)', help='amount of target sequence context')
    parser.add_argument('--dataset', type=str, default='off-target', help='which dataset to use')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode will run models eagerly')
    parser.add_argument('--fig_ext', type=str, default='.svg', help='which file extension to use when saving plots')
    parser.add_argument('--filter_method', type=str, default='MinActiveRatio', help='gene filtering method')
    parser.add_argument('--holdout', type=str, default='guides', help='how to assemble cross-validation folds')
    parser.add_argument('--indels', action='store_true', default=False, help='include targets with indels')
    parser.add_argument('--kwargs', type=str, default='{}', help='model hyper-parameters')
    # parser.add_argument('--kwargs', type=str, default="{'features': []}")
    parser.add_argument('--loss', type=str, default='log_cosh', help='training loss function')
    parser.add_argument('--min_active_ratio', type=float, default=0.13, help='ratio of active guides to keep a gene')
    parser.add_argument('--model', type=str, default='Tiger1D', help='model name')
    parser.add_argument('--normalization', type=str, default='FrequentistQuantile', help='normalization method')
    parser.add_argument('--nt_quantile', type=float, default=0.01, help='active guide non-targeting quantile threshold')
    parser.add_argument('--pm_only', action='store_true', default=False, help='use only perfect match guides')
    parser.add_argument('--use_guide_seq', action='store_true', default=False, help='use guide sequence for PM only')

    return parser


def parse_common_arguments(parser: argparse.ArgumentParser):

    # parse arguments
    args = parser.parse_args()

    # convert context string to the appropriate type
    if hasattr(args, 'context'):
        if args.context[0].isnumeric():
            args.context = int(args.context)
        elif args.context[0] == '(':
            args.context = tuple(int(x) for x in args.context[1:-1].split(','))
        else:
            raise NotImplementedError

    # if data-set only has PM guides, force PM-only flag
    if hasattr(args, 'dataset') and set(load_data(args.dataset)[0]['guide_type'].unique()) == {'PM'}:
        args.pm_only = True

    # process kwargs string descriptor into a dictionary that is usable by python
    if hasattr(args, 'kwargs'):
        args.kwargs = json.loads(args.kwargs.replace('\'', '\"'))
    for key, value in args.kwargs.items():
        if args.kwargs[key] == 'True':
            args.kwargs[key] = True
        if args.kwargs[key] == 'False':
            args.kwargs[key] = False
        if isinstance(args.kwargs[key], list):
            args.kwargs[key] = set(args.kwargs[key])

    # if dataset has mismatches guides, then guide sequence is required
    if not args.pm_only and set(load_data(args.dataset)[0]['guide_type'].unique()) != {'PM'}:
        args.use_guide_seq = True

    return args


def data_directory(pm_only: bool, indels: bool):
    if pm_only:
        return 'pm'
    elif indels:
        return 'indels'
    else:
        return 'no_indels'


def regression_metrics(target_lfc, predicted_lfc=None, predicted_label_likelihood=None):
    """
    Compute regression metrics
    :param target_lfc: target LFC
    :param predicted_lfc: predicted LFC
    :param predicted_label_likelihood: predicted label
    :return: Pearson/Spearman correlation coefficients and their standard errors
    """
    assert not (predicted_label_likelihood is None and predicted_lfc is None)

    # if predicted LFCs are absent, use negative label likelihood (a high label likelihood implies a more negative LFC)
    if predicted_lfc is None:
        predicted_lfc = -predicted_label_likelihood

    # remove NaNs if they exist
    i_nan = np.isnan(target_lfc) | np.isnan(predicted_lfc)
    target_lfc = target_lfc[~i_nan]
    predicted_lfc = predicted_lfc[~i_nan]

    # compute metrics and their standard errors
    n = len(target_lfc)
    pearson = pearsonr(target_lfc, predicted_lfc)[0]
    pearson_err = np.sqrt((1 - pearson ** 2) / (n - 2))
    spearman = spearmanr(target_lfc, predicted_lfc)[0]
    spearman_err = np.sqrt((1 - spearman ** 2) / (n - 2))

    return pearson, pearson_err, spearman, spearman_err


def roc_and_prc_from_lfc(target_label, predicted_label_likelihood=None, predicted_lfc=None):
    """
    Compute ROC and PRC points
    :param target_label: target response
    :param predicted_label_likelihood: predicted label
    :param predicted_lfc: predicted LFC
    :return: ROC and PRC
    """
    if target_label is None:
        return (None, None), (None, None)
    assert not (predicted_label_likelihood is None and predicted_lfc is None)

    # if predicted labels are absent, use negative LFC (a highly negative LFC implies a positive label)
    if predicted_label_likelihood is None:
        predicted_label_likelihood = -predicted_lfc

    # ROC and PRC values
    fpr, tpr, _ = roc_curve(target_label, predicted_label_likelihood)
    precision, recall, _ = precision_recall_curve(target_label, predicted_label_likelihood)

    return (fpr, tpr), (precision, recall)


def classification_metrics(target_label, predicted_label_likelihood=None, predicted_lfc=None, n_bootstraps=100):
    """
    Compute classification metrics
    :param target_label: target response
    :param predicted_label_likelihood: predicted label
    :param predicted_lfc: predicted LFC
    :param n_bootstraps: number of bootstraps used to estimate AUPRC standard error
    :return: AUROC, AUPRC
    """
    if target_label is None:
        return None, None, None, None

    # ROC and PRC
    (fpr, tpr), (precision, recall) = roc_and_prc_from_lfc(target_label, predicted_label_likelihood, predicted_lfc)

    # area under the above curves
    auroc = auc(fpr, tpr)
    auprc = auc(recall, precision)

    # compute AUROC standard errors
    if predicted_label_likelihood is None:
        predicted_label_likelihood = -predicted_lfc
    auroc_delong, auroc_var = delong_roc_variance(target_label.to_numpy().astype(float),
                                                  predicted_label_likelihood.to_numpy())
    auroc_err = auroc_var ** 0.5

    # bootstrap estimate AUPRC standard errors
    df_bootstrap = pd.DataFrame.from_dict({'target_label': target_label,
                                           'predicted_label_likelihood':predicted_label_likelihood})
    auprc_samples = np.empty(n_bootstraps)
    for n in range(n_bootstraps):
        df_sample = df_bootstrap.sample(len(df_bootstrap), replace=True)
        precision, recall, _ = precision_recall_curve(df_sample['target_label'],
                                                      df_sample['predicted_label_likelihood'])
        auprc_samples[n] = auc(recall, precision)
    auprc_err = auprc_samples.std()

    return auroc, auroc_err, auprc, auprc_err


def measure_performance(df_tap, index=None, silence=False):
    """
    Compute performance metrics over the provided predictions
    :param df_tap: DataFrame of targets and predictions
    :param index: an optional index for the returned DataFrame
    :param silence: whether to silence printing performance
    :return: DataFrame of performance metrics
    """
    # pack predictions into a dictionary (either can be None depending on model, but subsequent functions handle that)
    predictions = {'predicted_lfc': df_tap.get('predicted_lfc'),
                   'predicted_label_likelihood': df_tap.get('predicted_label_likelihood')}

    # compute metrics
    r, r_err, rho, rho_err = regression_metrics(df_tap['target_lfc'], **predictions)
    auroc, auroc_err, auprc, auprc_err = classification_metrics(df_tap.get('target_label'), **predictions)

    # generate ROC and PRC curves
    (fpr, tpr), (precision, recall) = roc_and_prc_from_lfc(df_tap.get('target_label'), **predictions)

    # pack performance into a dataframe
    df = pd.DataFrame({
        'Pearson': [r],
        'Pearson err': [r_err],
        'Spearman': [rho],
        'Spearman err': [rho_err],
        'AUROC': [auroc],
        'AUROC err': [auroc_err],
        'AUPRC': [auprc],
        'AUPRC err': [auprc_err],
        'ROC': [{'fpr': fpr, 'tpr': tpr}],
        'PRC': [{'precision': precision, 'recall': recall}]},
        index=index)

    # print performance
    cols = ['Pearson', 'Spearman', 'AUROC', 'AUPRC']
    if not silence:
        print(' | '.join([col + ': {:.4f}'.format(df.iloc[0][col]) for col in cols if df.iloc[0][col] is not None]))

    return df


def statistical_tests(reference_model, performance, predictions, n_bootstraps=100):
    """
    Compute p-values for performance metrics
    :param reference_model: pandas index of the reference model
    :param performance: pandas dataframe of performance metrics
    :param predictions: pandas dataframe of predictions
    :param n_bootstraps: number of bootstraps for AUPRC significance
    :return: modified performance with p-values added
    """
    # reference model bootstrap samples
    auprc_ref = np.empty(n_bootstraps)
    df_reference = predictions.loc[reference_model]
    for n in range(n_bootstraps):
        df_sample = df_reference.sample(len(df_reference), replace=True)
        precision, recall, _ = precision_recall_curve(df_sample['target_label'], -df_sample['predicted_lfc'])
        auprc_ref[n] = auc(recall, precision)

    # loop over alternative models
    for alternative_model in set(performance.index.unique()) - {reference_model}:

        # align targets and predictions for the two hypothesis
        df = pd.merge(predictions.loc[reference_model, ['guide_seq', 'target_label', 'predicted_lfc']],
                      predictions.loc[alternative_model, ['guide_seq', 'target_label', 'predicted_lfc']],
                      on=['guide_seq', 'target_label'], suffixes=('_ref', '_alt'))

        # loop over correlations
        for correlation, f_corr in [('Pearson', pearsonr), ('Spearman', spearmanr)]:

            # Steiger's test
            r1 = performance.loc[reference_model, correlation]
            r2 = performance.loc[alternative_model, correlation]
            r12 = abs(f_corr(df['predicted_lfc_ref'], df['predicted_lfc_alt'])[0])
            n = len(predictions)
            z1 = 0.5 * (np.log(1 + r1) - np.log(1 - r1))
            z2 = 0.5 * (np.log(1 + r2) - np.log(1 - r2))
            rm2 = (r1 ** 2 + r2 ** 2) / 2
            f = (1 - r12) / 2 / (1 - rm2)
            h = (1 - f * rm2) / (1 - rm2)
            z = abs(z1 - z2) * ((n - 3) / (2 * (1 - r12) * h)) ** 0.5
            log10_p = (norm.logcdf(-z) + np.log(2)) / np.log10(np.e)
            performance.loc[alternative_model, correlation + ' log10(p)'] = log10_p

        # DeLong's test
        log10_p = delong_roc_test(
            ground_truth=df['target_label'].to_numpy().astype(float),
            predictions_one=df['predicted_lfc_ref'].to_numpy(),
            predictions_two=df['predicted_lfc_alt'].to_numpy())
        performance.loc[alternative_model, 'AUROC log10(p)'] = log10_p[0][0]

        # bootstrap KS test
        auprc_alt = np.empty(n_bootstraps)
        for n in range(n_bootstraps):
            df_sample = df.sample(len(df), replace=True)
            precision, recall, _ = precision_recall_curve(df_sample['target_label'], -df_sample['predicted_lfc_alt'])
            auprc_alt[n] = auc(recall, precision)
        performance.loc[alternative_model, 'AUPRC log10(p)'] = np.log10(ks_2samp(auprc_ref, auprc_alt)[1])

    return performance


def measure_guide_type_performance(predictions, reference=None):
    """
    Compute performance metrics for each guide type
    :param predictions: DataFrame of targets and predictions
    :param reference: reference for statistical test (None will bypass those tests)
    :return: DataFrame of performance metrics
    """
    # guide type performance
    performance = pd.DataFrame()
    for guide_type in predictions['guide_type'].unique():
        df = predictions.loc[predictions.guide_type == guide_type, :]
        performance_guide_type = pd.DataFrame()
        indices = df.index.unique()
        for i in range(len(indices)):
            index = indices[i:i+1]
            performance_add = measure_performance(df.loc[index, :], index, silence=True)
            performance_guide_type = pd.concat([performance_guide_type, performance_add])
        performance_guide_type['guide_type'] = guide_type
        if reference is None:
            performance = pd.concat([performance, performance_guide_type])
        else:
            performance = pd.concat([performance, statistical_tests(reference, performance_guide_type, df)])

    return performance
