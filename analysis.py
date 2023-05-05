import os
import utils
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from data import label_and_filter_data, load_data, sequence_complement, SCALAR_FEATS
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

NUCLEOTIDES = ['A', 'C', 'T', 'G']
METRICS = ['Pearson', 'Spearman', 'AUROC', 'AUPRC']


def experiment_results_path(dataset: str, experiment: str, data_dir: str, holdout: str):
    return os.path.join('experiments', dataset, experiment, data_dir, holdout)


def plot_performance(performance: pd.DataFrame, predictions: pd.DataFrame, hue: str, null: str, title: str):

    # remove replicate performance and predictions
    replicates = performance.loc[performance.index.get_level_values(hue) == 'Replicates', :]
    performance = performance.loc[performance.index.get_level_values(hue) != 'Replicates', :]
    predictions = predictions.loc[predictions.index.get_level_values(hue) != 'Replicates', :]

    # check for statistically significant differences
    performance = utils.statistical_tests(null, performance, predictions)

    # initialize figure
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    fig.suptitle(title, wrap=True)

    # summary performance subplot
    metrics = performance[METRICS].reset_index().melt(id_vars=hue, var_name='Metric')
    ax[0].set_title('Summary Performance')
    sns.barplot(x='Metric', y='value', hue=hue, data=metrics, ax=ax[0], order=METRICS)
    for j, metric in enumerate(METRICS):
        for i, idx in enumerate(performance.index.unique()):
            x = ax[0].containers[i][j].get_x() + ax[0].containers[i][j].get_width() / 2
            y = performance.loc[idx, metric]
            error = performance.loc[idx, metric + ' err']
            ax[0].errorbar(x=x, y=y, yerr=2 * error, color='black')
            if idx == null:
                ax[0].text(x, y + 2 * error + 0.01, '$H_0$', horizontalalignment='center')
            elif performance.loc[idx, metric + ' log10(p)'] < np.log10(0.05):
                marker = '-' if performance.loc[null, metric] > y else '+'
                ax[0].text(x, y + 2 * error + 0.01, marker, horizontalalignment='center')
        if len(replicates) == 1:
            x = [ax[0].containers[0][j].get_x(), ax[0].containers[i][j].get_x() + ax[0].containers[i][j].get_width()]
            y = 2 * replicates[metric].values.tolist()
            ax[0].plot(x, y, alpha=0.5, color='gray')
    ax[0].set_ylim([0, 1])

    # ROC and PRC plots
    for i, curve in enumerate(['ROC', 'PRC']):
        if curve == 'ROC':
            ax[i + 1].set_title('Receiver Operating Characteristic')
            x = 'fpr'
            x_label = 'False Positive Rate'
            y = 'tpr'
            y_label = 'True Positive Rate'
        elif curve == 'PRC':
            ax[i + 1].set_title('Precision Recall')
            x = 'recall'
            x_label = 'Recall'
            y = 'precision'
            y_label = 'Precision'
        else:
            raise NotImplementedError
        for idx in performance.index.unique():
            ax[i + 1].plot(performance.loc[idx, curve][x], performance.loc[idx, curve][y], label=idx)
        if len(replicates) == 1:
            ax[i + 1].plot(replicates.iloc[0][curve][x], replicates.iloc[0][curve][y],
                           alpha=0.5, color='gray', label='Replicates')
        ax[i + 1].set_xlim([0, 1])
        ax[i + 1].set_ylim([0, 1])
        ax[i + 1].set_xlabel(x_label)
        ax[i + 1].set_ylabel(y_label)
        ax[i + 1].legend(title=hue)

    # add replicate legend to first plot
    legend = ax[0].legend(title='Model', loc='lower right')
    handles, labels = ax[0].get_legend_handles_labels()
    handles.append(ax[-1].get_legend_handles_labels()[0][-1])
    labels.append(ax[-1].get_legend_handles_labels()[1][-1])
    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())

    # make things pretty
    plt.tight_layout()

    return fig


def plot_performance_by_type(performance: pd.DataFrame, null: str, title: str):

    # guide order
    performance.rename(columns={'guide_type': 'Guide type'}, inplace=True)
    order = ['SM', 'DM', 'RDM', 'TM', 'RTM', 'SI', 'CI', 'DI', 'SD', 'CD', 'DD']
    [order.remove(gt) for gt in set(order) - set(performance['Guide type'].unique())]

    # drop index levels
    assert set(performance.index.names) == {'Model'}
    levels = performance.index.names + ['Guide type']
    performance.reset_index(inplace=True)

    # plot figure
    metrics = performance[METRICS + levels].melt(id_vars=levels, var_name='Metric')
    performance.set_index(['Model', 'Guide type'], inplace=True)
    g = sns.catplot(data=metrics, kind='bar', x='Guide type', y='value', order=order, hue='Model', col='Metric')
    g.fig.suptitle(title)
    sns.move_legend(g, 'upper right', title='Model')
    for metric, ax in g.axes_dict.items():
        for j, guide_type in enumerate(order):
            for i, model in enumerate(performance.index.get_level_values('Model').unique()):

                # error bars
                x = ax.containers[i][j].get_x() + ax.containers[i][j].get_width() / 2
                y = performance.loc[(model, guide_type), metric]
                error = performance.loc[(model, guide_type), metric + ' err']
                ax.errorbar(x=x, y=y, yerr=2 * error, color='black')

                # statistical significance
                if model == null:
                    ax.text(x, y + 2 * error + 0.01, '$H_0$', horizontalalignment='center')
                elif performance.loc[(model, guide_type), metric + ' log10(p)'] < np.log10(0.05):
                    marker = '-' if performance.loc[(null, guide_type), metric] > y else '+'
                    ax.text(x, y + 2 * error + 0.01, marker, horizontalalignment='center')

    return g.fig


def figure_save_path(dataset: str, experiment: str, data_dir: str, holdout: str):
    return os.path.join('figures', dataset, experiment, data_dir, holdout)


def save_fig(figure: plt.Figure, fig_path: str, file_name: str, file_ext: str):
    os.makedirs(fig_path, exist_ok=True)
    figure.savefig(os.path.join(fig_path, file_name + file_ext))


def drop_unused_index_levels(df: pd.DataFrame):
    for level in df.index.names:
        if len(df.index.unique(level)) == 1:
            df.set_index(df.index.droplevel(level), inplace=True)
    return df


def plot_label_and_filter_results(dataset: str, data_sub_dir: str, holdout: str, fig_ext: str):

    # load results, targets, and predictions
    exp_path = experiment_results_path(dataset, 'label-and-filter', data_sub_dir, holdout)
    try:
        performance = pd.read_pickle(os.path.join(exp_path, 'performance.pkl'))
        predictions = pd.read_pickle(os.path.join(exp_path, 'predictions.pkl'))
    except FileNotFoundError:
        return None

    # initialize figure
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    kwargs = {'linewidth': 2.5, 'marker': 'o', 'markersize': 7.5}

    # pull out relevant columns and check for a single configuration otherwise
    performance = performance.reset_index(['filter-method', 'non-targeting quantile', 'min active ratio'])
    predictions = predictions.reset_index(['filter-method', 'non-targeting quantile', 'min active ratio'])
    assert performance.index.nunique() == predictions.index.nunique() == 1

    # loop over the various NT quantiles that label guides as active or inactive
    for method in performance['filter-method'].unique():
        for nt_cutoff in performance['non-targeting quantile'].unique():

            # partial index
            perf_idx = (performance['filter-method'] == method) & (performance['non-targeting quantile'] == nt_cutoff)
            pred_idx = (predictions['filter-method'] == method) & (predictions['non-targeting quantile'] == nt_cutoff)

            # plot performance effects
            x = performance['min active ratio'].dropna().unique()
            label = method + ': ' + str(nt_cutoff)
            for i, metric in enumerate(METRICS):
                y = performance.loc[perf_idx, metric].values.tolist()
                y_err = performance.loc[perf_idx, metric + ' err'].values.tolist()
                if method == 'GoldStandard':
                    y = y * len(x)
                    y_err = y_err * len(x)
                ax[i // 2, i % 2].errorbar(x, y, 2 * y_err, label=label, elinewidth=1, **kwargs)
                ax[i // 2, i % 2].set_title(metric)

            # plot data utilization effects
            if method == 'GoldStandard':
                num_genes = [predictions.loc[pred_idx, 'gene'].nunique()] * len(x)
                num_guides = [len(predictions.loc[pred_idx])] * len(x)
            elif method == 'MinActiveRatio':
                num_genes = []
                num_guides = []
                for min_ratio in x:
                    idx = pred_idx & (predictions['min active ratio'] == min_ratio)
                    num_genes += [predictions.loc[idx, 'gene'].nunique()]
                    num_guides += [len(predictions.loc[idx])]
            else:
                raise NotImplementedError
            ax[0, 2].plot(x, num_genes, label=label, **kwargs)
            ax[0, 2].set_title('# of genes')
            ax[1, 2].plot(x, num_guides, label=label, **kwargs)
            ax[1, 2].set_title('# of guides')

    # finalize plot
    for a in ax.flatten():
        a.set_xlabel('Min. active ratio')
    ax[0, -1].legend(title='NT quantile', bbox_to_anchor=(1, 1), loc='upper left')
    ax[0, 2].set_ylim(bottom=0)
    ax[1, 2].set_ylim(bottom=0)
    plt.tight_layout()

    # save figure
    fig_path = figure_save_path(dataset, 'label-and-filter', data_sub_dir, holdout)
    save_fig(fig, fig_path, 'summary', fig_ext)


def plot_model_performances(dataset: str, data_sub_dir: str, holdout: str, fig_ext: str):

    # load results, targets, and predictions
    exp_path = experiment_results_path(dataset, 'model', data_sub_dir, holdout)
    try:
        performance = pd.read_pickle(os.path.join(exp_path, 'performance.pkl'))
        predictions = pd.read_pickle(os.path.join(exp_path, 'predictions.pkl'))
    except FileNotFoundError:
        return None

    # drop unused index levels
    performance = drop_unused_index_levels(performance).sort_index()
    predictions = drop_unused_index_levels(predictions).sort_index()

    # remove model from index
    index_names = list(performance.index.names)
    index_names.remove('model')
    performance = performance.reset_index().set_index(index_names)
    predictions = predictions.reset_index().set_index(index_names)

    # loop over the configurations
    for i, index in enumerate(performance.index.unique()):

        # plot performance
        df_perf = performance.loc[[index]].reset_index(drop=True).set_index('model')
        df_pred = predictions.loc[[index]].reset_index(drop=True).set_index('model')
        title = 'Model Performance: ' + str(dict(zip(performance.index.names, [index])))
        fig = plot_performance(df_perf, df_pred, hue='model', null=df_perf['Pearson'].idxmax(), title=title)

        # save figure
        fig_path = figure_save_path(dataset, 'model', data_sub_dir, holdout)
        save_fig(fig, fig_path, 'models_config' + str(i + 1), fig_ext)


def plot_sequence_context_performances(dataset: str, data_sub_dir: str, holdout: str, fig_ext: str):

    # load results, targets, and predictions
    exp_path = experiment_results_path(dataset, 'context', data_sub_dir, holdout)
    try:
        performance = pd.read_pickle(os.path.join(exp_path, 'performance.pkl'))
        predictions = pd.read_pickle(os.path.join(exp_path, 'predictions.pkl'))
    except FileNotFoundError:
        return None

    # remove context from indices
    index_names = list(performance.index.names)
    index_names.remove('context')
    performance = performance.reset_index().set_index(index_names).sort_index()
    predictions = predictions.reset_index().set_index(index_names).sort_index()

    # partition context
    performance['left context'] = performance['context'].apply(lambda c: c[0])
    performance['right context'] = performance['context'].apply(lambda c: c[1])
    predictions['left context'] = predictions['context'].apply(lambda c: c[0])
    predictions['right context'] = predictions['context'].apply(lambda c: c[1])

    # loop over unique indices
    for i, index in enumerate(performance.index.unique()):

        # plot version
        plot_kwargs = {'linewidth': 1.5, 'elinewidth': 1}
        fig_plot, ax_plot = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        fig_plot.suptitle('Configuration = ' + str(index), wrap=True)
        ax_plot = ax_plot.flatten()

        # matrix version
        fig_matrix, ax_matrix = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        fig_matrix.suptitle('Configuration = ' + str(index), wrap=True)
        ax_matrix = ax_matrix.flatten()

        # get metrics, targets, and predictions for this configuration
        performance_config = performance.loc[index].reset_index(drop=True).set_index(['left context', 'right context'])
        predictions_config = predictions.loc[index].reset_index(drop=True).set_index(['left context', 'right context'])
        performance_config = performance_config.sort_index()
        predictions_config = predictions_config.sort_index()

        # run statistical tests
        reference_model = performance_config['Pearson'].idxmax()
        performance_config = utils.statistical_tests(reference_model, performance_config, predictions_config)

        # loop over the metrics
        for j, metric in enumerate(METRICS):
            p_val = metric + ' log10(p)'

            # 5p context
            df = performance_config.loc[(slice(None), 0), [metric, metric + ' err']]
            ax_plot[j].errorbar(x=df.index.get_level_values('left context') - 0.2,
                                y=df[metric],
                                yerr=2 * df[metric + ' err'],
                                label='5p context',
                                **plot_kwargs)

            # 3p context
            df = performance_config.loc[(0, slice(None)), [metric, metric + ' err']]
            ax_plot[j].errorbar(x=df.index.get_level_values('right context'),
                                y=df[metric],
                                yerr=2 * df[metric + ' err'],
                                label='3p context',
                                **plot_kwargs)

            # symmetric context
            symmetric = performance_config.index.get_level_values('left context').values
            symmetric = (symmetric == performance_config.index.get_level_values('right context').values)
            df = performance_config.loc[symmetric, [metric, metric + ' err']]
            ax_plot[j].errorbar(x=df.index.get_level_values('right context') + 0.2,
                                y=df[metric],
                                yerr=2 * df[metric + ' err'],
                                label='5/3p context',
                                **plot_kwargs)

            # add title_legend
            ax_plot[j].set_title(metric)
            ax_plot[j].legend()

            try:
                df = performance_config[[metric, p_val]].copy()
                df['text'] = ''
                df.loc[df[p_val].isna(), 'text'] = '$H_0$'
                alternatives = ~df[p_val].isna() & (df[p_val] < np.log10(0.05))
                df.loc[alternatives & (df[metric] > df.loc[reference_model, metric]), 'text'] = '+'
                df.loc[alternatives & (df[metric] < df.loc[reference_model, metric]), 'text'] = '-'
                df.reset_index(inplace=True)
                df_annot = df.pivot('left context', 'right context', 'text')
            except KeyError:
                df = performance_config[['left context', 'right context'] + [metric]]
                df_annot = None
            df = df.pivot('left context', 'right context', metric)
            cbar_kws = dict(format='%4.3f', ticks=np.linspace(np.nanmin(df.values), np.nanmax(df.values), 5))
            sns.heatmap(df, annot=df_annot, ax=ax_matrix[j], center=df.loc[(0, 0)], cmap='coolwarm', fmt='', cbar_kws=cbar_kws)
            ax_matrix[j].set_title(metric)

        # save figure
        fig_path = figure_save_path(dataset, 'context', data_sub_dir, holdout)
        save_fig(fig_plot, fig_path, 'plot' + str(i + 1), fig_ext)

        # save figure
        fig_path = figure_save_path(dataset, 'context', data_sub_dir, holdout)
        save_fig(fig_matrix, fig_path, 'matrix' + str(i + 1), fig_ext)


def plot_non_sequence_feature_performances(dataset: str, data_sub_dir: str, holdout: str, fig_ext: str):

    # load results for both experiment parts
    exp_path_individual = experiment_results_path(dataset, 'feature-groups-individual', data_sub_dir, holdout)
    exp_path_cumulative = experiment_results_path(dataset, 'feature-groups-cumulative', data_sub_dir, holdout)
    try:
        performance_individual = pd.read_pickle(os.path.join(exp_path_individual, 'performance.pkl'))
        performance_cumulative = pd.read_pickle(os.path.join(exp_path_cumulative, 'performance.pkl'))
    except FileNotFoundError:
        return None

    # drop unused index levels
    performance_individual = drop_unused_index_levels(performance_individual)
    performance_cumulative = drop_unused_index_levels(performance_cumulative)
    assert {'features', 'feature group'} == set(performance_individual.index.names)
    assert {'features', 'feature group'} == set(performance_cumulative.index.names)
    performance_individual.reset_index('features', drop=True, inplace=True)
    performance_cumulative.reset_index('features', drop=True, inplace=True)

    # feature group ordering is that of the cumulative performances
    groups = performance_cumulative.index.values.tolist()

    # generate a figure for each metric
    kwargs = {'linewidth': 2.5, 'elinewidth': 1, 'marker': 'o', 'markersize': 7.5}
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.suptitle(metric + ' Feature Importance for ' + holdout[:-1].capitalize() + ' Holdouts')

        # individual performances
        ax.errorbar(x=np.arange(len(groups)),
                    y=performance_individual.loc[groups, metric],
                    yerr=2 * performance_individual.loc[groups, metric + ' err'],
                    color='tab:blue', ecolor='tab:blue',
                    label='Per-feature',
                    **kwargs)

        # cumulative performances
        ax.errorbar(x=np.arange(len(groups)) + 0.2,
                    y=performance_cumulative.loc[groups, metric],
                    yerr=2 * performance_cumulative.loc[groups, metric + ' err'],
                    color='tab:orange', ecolor='tab:orange',
                    label='Cumulative',
                    **kwargs)

        # clean up
        ax.legend(loc='upper left')
        ax.set_xticks(np.arange(len(groups)))
        ax.xaxis.set_ticklabels(groups, rotation=20, ha='right')
        ax.set_ylabel(metric)
        plt.tight_layout()

        # save figure
        fig_path = figure_save_path(dataset, 'feature-groups', data_sub_dir, holdout)
        save_fig(fig, fig_path, metric, fig_ext)


def plot_learning_curves(dataset: str, data_sub_dir: str, holdout: str, fig_ext: str):

    # load performances
    exp_path = experiment_results_path(dataset, 'learning-curve', data_sub_dir, holdout)
    try:
        df_performance = pd.read_pickle(os.path.join(exp_path, 'performance.pkl'))
    except FileNotFoundError:
        return None

    # drop unused index levels
    df_performance = drop_unused_index_levels(df_performance)
    assert set(df_performance.index.names) == {'training utilization'}

    # plot learning curve
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.suptitle('Learning curve: ' + args.holdout)
    kwargs = {'linewidth': 2.5, 'elinewidth': 1, 'marker': 'o', 'markersize': 5}
    for i, metric in enumerate(['Pearson', 'Spearman', 'AUROC', 'AUPRC']):
        ax[i // 2, i % 2].errorbar(x=df_performance.index.values,
                                   y=df_performance[metric],
                                   yerr=2 * df_performance[metric + ' err'],
                                   color='tab:blue', ecolor='tab:blue',
                                   **kwargs)
        ax[i // 2, i % 2].set_xlabel('Data Utilization')
        ax[i // 2, i % 2].set_title(metric)
    plt.tight_layout()

    # save figure
    fig_path = figure_save_path(dataset, 'learning-curve', data_sub_dir, holdout)
    save_fig(fig, fig_path, 'learning_curve', fig_ext)


def sequence_mask(df_shap: pd.DataFrame, guide_nt: str, target_nt: str):
    # find locations where guide and target NTs match specified values
    target_mask = np.array(df_shap['target_seq'].apply(lambda seq: [s == target_nt for s in seq]).to_list())
    if 'guide_seq' in df_shap.columns:
        guide_mask = np.array(df_shap['guide_seq'].apply(lambda seq: [s == guide_nt for s in seq]).to_list())
    else:
        guide_mask = np.ones_like(target_mask)

    return guide_mask * target_mask


def sequence_shap(df_shap: pd.DataFrame, guide_nts: str = '', target_nts: str = ''):
    # determine guide length and context amount
    assert df_shap['target_seq'].apply(len).nunique() == df_shap['target:A'].apply(len).nunique() == 1
    guide_len = df_shap['target_seq'].apply(len).unique()[0]  # target sequence does not include context
    context = (df_shap['target:A'].apply(len).unique()[0] - df_shap['target_seq'].apply(len).unique()[0]) // 2

    # add guide and target SHAP values for all respectively listed nucleotides
    guide_shap = np.zeros([len(df_shap), guide_len])
    target_shap = np.zeros([len(df_shap), guide_len])
    for nt in NUCLEOTIDES:
        if nt in guide_nts:
            # guide is padded
            if set(df_shap['guide:' + nt].apply(len)) == set(df_shap['target:' + nt].apply(len).unique()):
                guide_shap += np.array(df_shap['guide:' + nt].to_list())[:, context: context + guide_len]
            # guide is not padded
            elif set(df_shap['guide:' + nt].apply(len)) == set(df_shap['target_seq'].apply(len).unique()):
                guide_shap += np.array(df_shap['guide:' + nt].to_list())
        if nt in target_nts:
            target_shap += np.array(df_shap['target:' + nt].to_list())[:, context: context + guide_len]

    return guide_shap + target_shap


def pearson_and_shap(data: pd.DataFrame, df_shap: pd.DataFrame, mode: str):
    assert mode in {'matches', 'mismatches'}

    # loop over guide-target base pairs according to requested mode
    df = pd.DataFrame()
    for target_nt in NUCLEOTIDES:
        guide_match = sequence_complement(target_nt)
        for guide_actual in [guide_match] if mode == 'matches' else set(NUCLEOTIDES) - {guide_match}:

            # Pearson(1[(guide,target) at i`th position], LFC)
            mask = sequence_mask(data, guide_actual, target_nt)
            r = np.empty(mask.shape[1])
            for i in range(len(r)):
                r[i] = pearsonr(mask[:, i], data['target_lfc'])[0]

            # E[SHAP | (guide,target) at i`th position]
            mask = sequence_mask(df_shap, guide_actual, target_nt)
            shap = (mask * sequence_shap(df_shap, guide_nts='ACGT', target_nts='ACGT')).sum(0) / mask.sum(0)

            # append results
            df = pd.concat([df, pd.DataFrame(
                data={'Pearson': r, 'SHAP': shap},
                index=pd.MultiIndex.from_arrays(
                    arrays=[np.arange(1, len(shap) + 1),
                            [guide_actual + target_nt] * len(shap),
                            [guide_match] * len(shap),
                            [guide_actual] * len(shap)],
                    names=('Position', 'Guide-Target', 'Guide PM', 'Guide MM')))])

    return df


def plot_sequence_match_effects(dataset: str, data_sub_dir: str, holdout: str, fig_ext: str):
    # load and label data for perfect matches
    data, data_nt = load_data(dataset=dataset, pm_only=True)
    data = label_and_filter_data(data, data_nt)
    assert data['guide_seq'].apply(len).nunique() == 1

    # load SHAP values for perfect matches
    exp_path = experiment_results_path(dataset, 'SHAP', data_sub_dir, holdout)
    try:
        df_shap = pd.read_pickle(os.path.join(exp_path, 'shap.pkl'))
    except FileNotFoundError:
        return None
    df_shap = df_shap[df_shap['guide_type'] == 'PM']

    # get pearson and shap values
    df = pearson_and_shap(data, df_shap, mode='matches')

    # print agreement between Pearson and SHAP values
    print('Agreement between Pearson and SHAP for matches = {:.4f}'.format(pearsonr(df['Pearson'], df['SHAP'])[0]))

    # plot results
    df.reset_index(inplace=True)
    guide_len = df['Position'].max()
    values = ('Pearson', 'SHAP')
    fig, ax = plt.subplots(ncols=len(values), figsize=(len(values) * 5, 5))
    fig.suptitle('Pearson and mean SHAP values for complementary guide-target base pairs')
    for i, value in enumerate(values):
        ax[i].plot(np.arange(1, guide_len + 1), np.zeros(guide_len), color='black', linestyle='--')
        sns.lineplot(x='Position', y=value, hue='Guide-Target', data=df, ax=ax[i])
        ax[i].legend(title='Guide-Target', bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()

    # save figure
    fig_path = figure_save_path(dataset, 'SHAP', data_sub_dir, holdout)
    save_fig(fig, fig_path, 'seq-match-effects', fig_ext)


def plot_sequence_mismatch_effects(dataset: str, data_sub_dir: str, holdout: str, fig_ext: str):
    # load and label data for mismatches
    data, data_nt = load_data(dataset=dataset, pm_only=False)
    data = label_and_filter_data(data, data_nt)
    data = data[data['guide_type'] != 'PM']
    if len(data) == 0:
        return

    # load SHAP values for mismatches
    exp_path = experiment_results_path(dataset, 'SHAP', data_sub_dir, holdout)
    try:
        df_shap = pd.read_pickle(os.path.join(exp_path, 'shap.pkl'))
    except FileNotFoundError:
        return None
    df_shap = df_shap[df_shap['guide_type'] != 'PM']
    if len(df_shap) == 0:
        return

    # get pearson and shap values
    df = pearson_and_shap(data, df_shap, mode='mismatches')

    # print agreement between Pearson and SHAP values
    df_test = df.dropna()
    agreement = pearsonr(df_test['Pearson'], df_test['SHAP'])[0]
    print('Agreement between Pearson and SHAP for mismatches = {:.4f}'.format(agreement))

    # mutation type column
    df.reset_index(inplace=True)
    df['Guide Mutation'] = df['Guide PM'] + '->' + df['Guide MM']
    df.sort_values(by=['Guide Mutation'], ascending=True, inplace=True)

    # plot mismatch effects
    guide_len = df['Position'].max()
    values = ('Pearson', 'SHAP')
    fig, ax = plt.subplots(nrows=len(values), ncols=4, figsize=(4 * 5, len(values) * 5))
    fig.suptitle('Pearson and mean SHAP values for guide mutations')
    for i, value in enumerate(values):
        for j, pm in enumerate(NUCLEOTIDES):
            ax[i, j].plot(np.arange(1, guide_len + 1), np.zeros(guide_len), color='black', linestyle='--')
            sns.lineplot(x='Position', y=value, hue='Guide Mutation', data=df[df['Guide PM'] == pm], ax=ax[i, j])
            ax[i, j].legend()
            ax[i, j].set_ylim(bottom=df[value].min() * 1.05, top=df[value].max() * 1.05)
    plt.tight_layout()

    # save figure
    fig_path = figure_save_path(dataset, 'SHAP', data_sub_dir, holdout)
    save_fig(fig, fig_path, 'seq-mismatch-effects', fig_ext)


def plot_non_sequence_pearson(dataset: str, data_sub_dir: str, holdout: str, fig_ext: str, pm_only: bool):

    # load and label data
    data, data_nt = load_data(dataset=dataset, pm_only=pm_only)
    if not pm_only and set(data['guide_type'].unique()) == {'PM'}:
        return
    data = label_and_filter_data(data, data_nt)

    # consider all available features
    available_features = list(set(SCALAR_FEATS).intersection(set(data.columns)))

    # remove any features with a constant value
    i = 0
    while i < len(available_features):
        if data[available_features[i]].nunique() == 1:
            available_features.pop(i)
        else:
            i += 1

    # non-sequence feature Pearson values
    r = np.empty(len(available_features))
    for i, feature in enumerate(available_features):
        r[i] = pearsonr(data[feature], data['target_lfc'])[0]

    # plot correlation values in descending order
    fig, ax = plt.subplots()
    fig.suptitle('Non-sequence features\' correlation values for ' + ('PM' if pm_only else 'all') + ' guides')
    df = pd.DataFrame({'Feature': available_features, 'Pearson': r})
    df.sort_values(by=['Pearson'], ascending=False, inplace=True)
    df['Effect'] = df['Pearson'].apply(lambda x: '+' if x >= 0 else '-')
    sns.barplot(x='Feature', y='Pearson', hue='Effect', data=df, ax=ax)
    ax.xaxis.set_ticklabels(df['Feature'], rotation=30, ha='right')
    ax.legend(title='Effect', bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()

    # save figure
    fig_path = figure_save_path(dataset, 'SHAP', data_sub_dir, holdout)
    save_fig(fig, fig_path, 'nonseq-pearson-match' if pm_only else 'nonseq-pearson-all', fig_ext)


def plot_non_sequence_effects(dataset: str, data_sub_dir: str, holdout: str, fig_ext: str, pm_only: bool):
    # load SHAP values
    exp_path = experiment_results_path(dataset, 'SHAP', data_sub_dir, holdout)
    try:
        df_shap = pd.read_pickle(os.path.join(exp_path, 'shap.pkl'))
    except FileNotFoundError:
        return None
    if pm_only:
        df_shap = df_shap[df_shap['guide_type'] == 'PM']
    elif set(df_shap['guide_type'].unique()) == {'PM'}:
        return
    if len(df_shap) == 0:
        return

    # index columns
    index_cols = ['target_seq', 'guide_seq']

    # SHAP values
    available_features = list(set(SCALAR_FEATS).intersection(set(df_shap.columns)))
    df_shap = df_shap[index_cols + available_features]
    order = df_shap[available_features].var(axis=0).sort_values(ascending=False).index.values.tolist()

    # input values
    df_data, _ = load_data(dataset, scale_non_seq_feats=True)
    df_data = df_data[index_cols + available_features]

    # join SHAP and input values
    df_shap = df_shap.melt(id_vars=index_cols, var_name='feature', value_name='SHAP').set_index(index_cols)
    df_data = df_data.melt(id_vars=index_cols, var_name='feature', value_name='value').set_index(index_cols)
    data = df_shap.set_index('feature', append=True).join(df_data.set_index('feature', append=True))
    data = data.reset_index('feature').reset_index(drop=True)
    data['x'] = data['feature'].apply(lambda f: order.index(f)) + np.random.uniform(-0.25, 0.25, size=len(data))

    # plot correlation values in descending order
    fig, ax = plt.subplots()
    fig.suptitle('Non-sequence features\' SHAP values for ' + ('PM' if pm_only else 'all') + ' guides')
    ax.scatter(x=data['x'], y=data['SHAP'], s=3, c=data['value'], alpha=0.1, rasterized=True)
    ax.set_xticks(list(range(len(order))), order, rotation=30, ha='right')
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1)), ax=ax, alpha=1)
    plt.tight_layout()

    # save figure
    fig_path = figure_save_path(dataset, 'SHAP', data_sub_dir, holdout)
    save_fig(fig, fig_path, 'nonseq-effects-match' if pm_only else 'nonseq-effects-all', fig_ext)


def cell_type_correction(df_cell_x, df_cell_y, values):

    # find common points
    idx_cols = ['gene', 'guide_seq']
    data_common = pd.concat([df_cell_x.set_index(idx_cols), df_cell_y.set_index(idx_cols)])
    data_common = data_common.pivot(columns=['cell'], values=values)

    # plot common and the full marginals
    cell_x = df_cell_x['cell'].unique()[0]
    cell_y = df_cell_y['cell'].unique()[0]
    g = sns.jointplot(data=data_common, x=cell_x, y=cell_y, joint_kws=dict(alpha=0.5))

    # fit cell type correction
    data_common.dropna(inplace=True)
    p = np.polyfit(x=data_common[cell_x], y=data_common[cell_y], deg=2)
    x = np.linspace(data_common[cell_x].min(), data_common[cell_y].max())
    g.ax_joint.plot(x, np.polyval(p, x), label='Correction')
    g.ax_joint.plot(x, x, color='black', linestyle=':')
    g.figure.suptitle('Proliferation Correction')

    return p, g.fig


if __name__ == '__main__':

    # ensure text is text in images
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['svg.fonttype'] = 'none'

    # parser arguments
    parser = utils.common_parser_arguments()
    args = utils.parse_common_arguments(parser)

    # generate and save plots
    data_sub_directory = utils.data_directory(args.pm_only, args.indels)
    plot_label_and_filter_results(args.dataset, data_sub_directory, args.holdout, args.fig_ext)
    plot_model_performances(args.dataset, data_sub_directory, args.holdout, args.fig_ext)
    plot_sequence_context_performances(args.dataset, data_sub_directory, args.holdout, args.fig_ext)
    plot_non_sequence_feature_performances(args.dataset, data_sub_directory, args.holdout, args.fig_ext)
    plot_learning_curves(args.dataset, data_sub_directory, args.holdout, args.fig_ext)
    plot_sequence_match_effects(args.dataset, data_sub_directory, args.holdout, args.fig_ext)
    plot_sequence_mismatch_effects(args.dataset, data_sub_directory, args.holdout, args.fig_ext)
    # plot_non_sequence_pearson(args.dataset, data_sub_directory, args.holdout, args.fig_ext, pm_only=True)
    # plot_non_sequence_pearson(args.dataset, data_sub_directory, args.holdout, args.fig_ext, pm_only=False)
    plot_non_sequence_effects(args.dataset, data_sub_directory, args.holdout, args.fig_ext, pm_only=True)
    plot_non_sequence_effects(args.dataset, data_sub_directory, args.holdout, args.fig_ext, pm_only=False)

    # show them
    plt.show()
