import argparse
import json
import numpy as np
import pandas as pd
from data import load_data
from scipy.stats import ks_2samp, norm, pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_curve, roc_auc_score
from roc_comparison.compare_auc_delong_xu import delong_roc_test, delong_roc_variance


def string_to_dict(dict_string: str):
    kwargs = json.loads(dict_string.replace('\'', '\"'))
    for key, value in kwargs.items():
        if kwargs[key] == 'True':
            kwargs[key] = True
        if kwargs[key] == 'False':
            kwargs[key] = False
        if isinstance(kwargs[key], list):
            kwargs[key] = set(kwargs[key])
    return kwargs


def common_parser_arguments():

    # common arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='tensorflow batch size')
    parser.add_argument('--context', type=str, default=None, help='amount of target sequence context')
    parser.add_argument('--dataset', type=str, default='off-target', help='which dataset to use')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode will run models eagerly')
    parser.add_argument('--fig_ext', type=str, default='.pdf', help='which file extension to use when saving plots')
    parser.add_argument('--filter_method', type=str, default=None, help='gene filtering method')
    parser.add_argument('--holdout', type=str, default='targets', help='how to assemble cross-validation folds')
    parser.add_argument('--indels', action='store_true', default=False, help='include targets with indels')
    parser.add_argument('--kwargs', type=str, default=None, help='model hyper-parameters')
    parser.add_argument('--loss', type=str, default='log_cosh', help='training loss function')
    parser.add_argument('--min_active_ratio', type=float, default=None, help='ratio of active guides to keep a gene')
    parser.add_argument('--model', type=str, default=None, help='model name')
    parser.add_argument('--normalization', type=str, default=None, help='normalization method')
    parser.add_argument('--normalization_kwargs', type=str, default=None, help='normalization parameters')
    parser.add_argument('--nt_quantile', type=float, default=None, help='active guide non-targeting quantile threshold')
    parser.add_argument('--pm_only', action='store_true', default=False, help='use only perfect match guides')
    parser.add_argument('--use_guide_seq', action='store_true', default=False, help='use guide sequence for PM only')
    parser.add_argument('--seed', type=int, default=None, help='random number seed')
    parser.add_argument('--seq_only', action='store_true', default=False, help='sequence only model')

    return parser


def parse_common_arguments(parser: argparse.ArgumentParser, dataset_parameter_override: str = None):

    # parse arguments
    args = parser.parse_args()

    # dataset-specific defaults
    dataset = dataset_parameter_override or args.dataset
    if dataset in {'off-target', 'flow-cytometry'}:
        args.context = args.context or '(1,1)'
        args.filter_method = args.filter_method or 'NoFilter'
        args.kwargs = args.kwargs or '{}'  # "{'features': []}"
        args.min_active_ratio = args.min_active_ratio or 0.1
        args.model = args.model or 'Tiger2D'
        args.normalization = args.normalization or 'FrequentistQuantile'
        args.normalization_kwargs = args.normalization_kwargs or "{'q_loc': 50, 'q_neg': 10, 'q_pos': 90}"
        args.nt_quantile = args.nt_quantile or 0.01
    elif 'junction' in dataset:
        args.context = args.context or '(1,1)'
        args.filter_method = args.filter_method or 'MinActiveRatio'
        args.kwargs = args.kwargs or '{}'  # "{'features': []}"
        args.min_active_ratio = args.min_active_ratio or 0.05
        args.model = args.model or 'Tiger1D'
        args.normalization = args.normalization or 'No'
        args.normalization_kwargs = args.normalization_kwargs or "{}"
        args.nt_quantile = args.nt_quantile or 0.01
    else:
        assert args.context is not None
        assert args.filter_method is not None
        assert args.kwargs is not None
        assert args.min_active_ratio is not None
        assert args.model is not None
        assert args.nt_quantile is not None

    # convert context string to the appropriate type
    if hasattr(args, 'context'):
        if args.context[0].isnumeric():
            args.context = int(args.context)
        elif args.context[0] == '(':
            args.context = tuple(int(x) for x in args.context[1:-1].split(','))
        else:
            raise NotImplementedError

    # check if dataset only contains PM guides
    pm_only = set(load_data(args.dataset)[0]['guide_type'].unique()) == {'PM'}

    # if dataset only has PM guides, force PM-only flag
    if hasattr(args, 'dataset') and pm_only:
        args.pm_only = True

    # process kwargs string descriptor into a dictionary that is usable by python
    if hasattr(args, 'kwargs'):
        args.kwargs = string_to_dict(args.kwargs)
    if hasattr(args, 'normalization_kwargs'):
        args.normalization_kwargs = string_to_dict(args.normalization_kwargs)

    # if dataset has mismatches guides, then guide sequence is required
    if not args.pm_only and not pm_only:
        args.use_guide_seq = True

    return args


def data_directory(pm_only: bool, indels: bool, seq_only: bool):
    if pm_only:
        directory = 'pm'
    elif indels:
        directory = 'indels'
    else:
        directory = 'no_indels'
    if seq_only:
        directory += '-seq_only'
    return directory


def total_least_squares_slope(x: np.array, y: np.array):
    """
    1-dimensional total least squares slope
    :param x: independent univariate variable
    :param y: dependent univariate variable
    :return: slope
    """
    u, s, v = np.linalg.svd(np.stack([x, y], axis=-1), full_matrices=False)
    return -v[1, 0] / v[1, 1]


def titration_ratio(df: pd.DataFrame, num_top_guides: int, correction: bool = False, transpose: bool =False):

    # total least squares slope correction
    if correction:
        x_name = 'predicted_lfc'
        y_name = 'observed_lfc'
        for guide_type in {'PM', 'SM'}:
            x = df.loc[df.guide_type == guide_type, x_name].to_numpy()
            y = df.loc[df.guide_type == guide_type, y_name].to_numpy()
            slope = total_least_squares_slope(x, y)
            df.loc[df.guide_type == guide_type, x_name] = slope * df.loc[df.guide_type == guide_type, x_name]

    # keep PM and SM guides for high confidence essential genes only
    essential_genes = df.loc[df.guide_type == 'PM'].groupby('gene')['observed_label'].sum()
    essential_genes = essential_genes.index.values[essential_genes > 6]
    df = df.loc[df.guide_type.isin({'PM', 'SM'}) & df.gene.isin(essential_genes)].copy()

    # only titer the top guides per transcript
    targets = []
    for gene in df['gene'].unique():
        index = (df.gene == gene) & (df.guide_type == 'PM')
        predictions = df.loc[index, 'predicted_lfc'].sort_values(ascending=not transpose).to_numpy()
        threshold = predictions[min(num_top_guides - 1, len(predictions) - 1)]
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
        df.loc[index, 'Observed ratio'] = df.loc[index, 'observed_lfc'].rank(ascending=False, pct=True)
        z = df.loc[index & (df.guide_type == 'PM'), 'Observed ratio'].to_numpy()
        df.loc[index, 'Observed ratio'] /= z

        # predicted titration
        df.loc[df.target_seq == target, 'Predicted ratio'] = 0
        index = (df.target_seq == target)
        df.loc[index, 'Predicted ratio'] = df.loc[index, 'predicted_lfc'].rank(ascending=transpose, pct=True)
        z = df.loc[index & (df.guide_type == 'PM'), 'Predicted ratio'].to_numpy()
        df.loc[index, 'Predicted ratio'] /= z

    return df


def regression_metrics(observations: pd.Series, predictions: pd.Series):
    """
    Compute regression metrics
    :param observations: observed variable
    :param predictions: predicted variable
    :return: Pearson/Spearman correlation coefficients and their standard errors
    """
    # remove NaNs if they exist
    i_nan = np.isnan(observations) | np.isnan(predictions)
    observed_lfc = observations[~i_nan]
    predicted_lfc = predictions[~i_nan]

    # compute calibration metrics
    slope = total_least_squares_slope(predicted_lfc, observed_lfc)

    # compute metrics and their standard errors
    n = len(observed_lfc)
    pearson = pearsonr(observed_lfc, predicted_lfc)[0]
    pearson_err = np.sqrt((1 - pearson ** 2) / (n - 2))
    spearman = spearmanr(observed_lfc, predicted_lfc)[0]
    spearman_err = np.sqrt((1 - spearman ** 2) / (n - 2))

    return slope, pearson, pearson_err, spearman, spearman_err


def roc_and_prc_from_lfc(observed_label: pd.Series, predictions: pd.Series):
    """
    Compute ROC and PRC points
    :param observed_label: observed response
    :param predictions: predictions
    :return: ROC and PRC
    """
    if observed_label is None or len(np.unique(observed_label)) != 2:
        return (None, None), (None, None)

    # ensure predictions positively correlate with labels or metrics will break (e.g. LFC needs to be sign flipped)
    predictions = np.sign(pearsonr(predictions, observed_label)[0]) * predictions

    # ROC and PRC values
    fpr, tpr, _ = roc_curve(observed_label, predictions)
    precision, recall, _ = precision_recall_curve(observed_label, predictions)

    return (fpr, tpr), (precision, recall)


def classification_metrics(observed_label: pd.Series, predictions: pd.Series, n_bootstraps=100):
    """
    Compute classification metrics
    :param observed_label: observed response
    :param predictions: predictions
    :param n_bootstraps: number of bootstraps used to estimate AUPRC standard error
    :return: AUROC, AUPRC
    """
    if observed_label is None or len(np.unique(observed_label)) != 2:
        return None, None, None, None

    # ROC and PRC
    (fpr, tpr), (precision, recall) = roc_and_prc_from_lfc(observed_label, predictions)

    # area under the above curves
    auroc = auc(fpr, tpr)
    auprc = auc(recall, precision)

    # compute AUROC standard errors
    predictions = np.sign(pearsonr(predictions, observed_label)[0]) * predictions
    auroc_delong, auroc_var = delong_roc_variance(observed_label.to_numpy().astype(float), predictions.to_numpy())
    auroc_err = auroc_var ** 0.5

    # bootstrap estimate AUPRC standard errors
    df_bootstrap = pd.DataFrame.from_dict({'observed_label': observed_label, 'predictions': predictions})
    auprc_samples = np.empty(n_bootstraps)
    for n in range(n_bootstraps):
        df_sample = df_bootstrap.sample(len(df_bootstrap), replace=True)
        precision, recall, _ = precision_recall_curve(df_sample['observed_label'], df_sample['predictions'])
        auprc_samples[n] = auc(recall, precision)
    auprc_err = auprc_samples.std()

    return auroc, auroc_err, auprc, auprc_err


def measure_performance(df: pd.DataFrame, index=None, obs_var='observed_lfc', pred_var='predicted_lfc', silence=False):
    """
    Compute performance metrics over the provided predictions
    :param df: DataFrame of targets and predictions
    :param index: an optional index for the returned DataFrame
    :param obs_var: observed variable name
    :param pred_var: predicted variable name
    :param silence: whether to silence printing performance
    :return: DataFrame of performance metrics
    """
    # compute metrics
    slope, r, r_err, rho, rho_err = regression_metrics(df[obs_var], df[pred_var])
    auroc, auroc_err, auprc, auprc_err = classification_metrics(df.get('observed_label'), df[pred_var])

    # generate ROC and PRC curves
    (fpr, tpr), (precision, recall) = roc_and_prc_from_lfc(df.get('observed_label'), df[pred_var])

    # pack performance into a dataframe
    df = pd.DataFrame({
        'Slope': [slope],
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
    # loop over alternative models
    for alternative_model in set(performance.index.unique()) - {reference_model}:

        # align targets and predictions for the two hypothesis
        df = pd.merge(predictions.loc[reference_model, ['guide_seq', 'predicted_lfc']],
                      predictions.loc[alternative_model, ['guide_seq', 'predicted_lfc']],
                      on=['guide_seq'], suffixes=('_ref', '_alt'))

        # Steiger's test for Pearson and Spearman correlations
        for correlation, f_corr in [('Pearson', pearsonr), ('Spearman', spearmanr)]:
            if correlation in performance.columns:
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

    # do we have observed activity labels
    if 'observed_label' in predictions.columns:

        # reference model AUPRC bootstrap samples
        auprc_ref = np.empty(n_bootstraps)
        df_reference = predictions.loc[reference_model]
        for n in range(n_bootstraps):
            df_sample = df_reference.sample(len(df_reference), replace=True)
            precision, recall, _ = precision_recall_curve(df_sample['observed_label'], -df_sample['predicted_lfc'])
            auprc_ref[n] = auc(recall, precision)

        # loop over alternative models
        for alternative_model in set(performance.index.unique()) - {reference_model}:

            # align targets and predictions for the two hypothesis
            df = pd.merge(predictions.loc[reference_model, ['guide_seq', 'observed_label', 'predicted_lfc']],
                          predictions.loc[alternative_model, ['guide_seq', 'observed_label', 'predicted_lfc']],
                          on=['guide_seq', 'observed_label'], suffixes=('_ref', '_alt'))

            # DeLong's AUROC test
            log10_p = delong_roc_test(
                ground_truth=df['observed_label'].to_numpy().astype(float),
                predictions_one=df['predicted_lfc_ref'].to_numpy(),
                predictions_two=df['predicted_lfc_alt'].to_numpy())
            performance.loc[alternative_model, 'AUROC log10(p)'] = log10_p[0][0]

            # bootstrap AUPRC KS test
            auprc_alt = np.empty(n_bootstraps)
            for n in range(n_bootstraps):
                df_sample = df.sample(len(df), replace=True)
                p, r, _ = precision_recall_curve(df_sample['observed_label'], -df_sample['predicted_lfc_alt'])
                auprc_alt[n] = auc(r, p)
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
