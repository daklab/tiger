import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from analysis import plot_performance, sequence_pearson_and_shap, save_fig, METRICS
from data import load_data, label_and_filter_data, sequence_complement
from junction_utils import construct_junctions, append_rbp_predictions
from matplotlib import pyplot as plt


def plot_training_set_differences(fig_path: str, fig_ext: str):

    # load performance
    base_dir = os.path.join('predictions', 'junction')
    loop_vars = [('junction', False)]
    loop_vars += list(itertools.product(['off-target', 'combined'], [False, True]))
    performance = pd.DataFrame()
    for dataset, corrected in loop_vars:
        file_name = dataset + ('-corrected' if corrected else '') + '-performance.pkl'
        if os.path.exists(os.path.join(base_dir, file_name)):
            df = pd.read_pickle(os.path.join(base_dir, file_name))
            df['Training set'] = dataset + (' (corrected)' if corrected else '')
            performance = pd.concat([performance, df])

    # if nothing was loaded, return
    if len(performance) == 0:
        return

    # plot performance
    order = list(performance.loc['junction'].groupby('Training set').mean().sort_values('AUPRC').index.values)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.suptitle('Training set performance impact')
    sns.barplot(data=performance.reset_index(), x='Training set', y='AUPRC', ax=ax, order=order, hue='dataset')
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment('right')
        tick.set_rotation(30)
        sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    save_fig(fig, fig_path, 'dataset_differences', fig_ext)


def plot_two_stage_model_performances(fig_path: str, fig_ext: str):

    # load performance
    try:
        performance = os.path.join('experiments', 'junction', 'tiger-team', 'pm', 'guides', 'performance.pkl')
        performance = pd.read_pickle(performance)
    except FileNotFoundError:
        return

    # plot summary results
    fig, ax = plt.subplots(figsize=(10, 5))
    df = performance[METRICS].melt(ignore_index=False, var_name='metric').reset_index()
    df['value'] = df['value'].astype(float)
    sns.barplot(x='metric', y='value', hue='task', data=df, order=METRICS, ax=ax)
    for i, task in enumerate([legend_text.get_text() for legend_text in ax.get_legend().get_texts()]):
        for j, metric in enumerate(METRICS):
            x = ax.containers[i][j].get_x() + ax.containers[i][j].get_width() / 2
            y = performance.loc[task, metric]
            yerr = performance.loc[task, metric + ' err']
            ax.errorbar(x=x, y=y, yerr=yerr, color='black')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    save_fig(fig, fig_path, 'two_stage_model', fig_ext)


def plot_target_vs_junction_shap(fig_path: str, fig_ext: str, min_overlap: int = 3):
    assert min_overlap > 0

    # load SHAP values
    try:
        shap = os.path.join('experiments', 'junction', 'tiger-team', 'pm', 'guides', 'shap.pkl')
        shap = pd.read_pickle(shap)
    except FileNotFoundError:
        return

    # load data and create junction dataset
    data = label_and_filter_data(*load_data('junction'))
    data_junc = data.copy()
    data_junc['target_seq'] = data['5p_context'] + data['target_seq'] + data['3p_context']
    data_junc['guide_seq'] = data_junc['target_seq'].apply(sequence_complement)

    # shapes
    guide_len = data['guide_seq'].apply(len).unique()[0]
    num_tiles = data['junction_olap_5p'].nunique()
    tile_start = data_junc['target_seq'].apply(len).unique()[0] // 2 + int(data['junction_olap_5p'].min() * guide_len)

    # prepare SHAP figure
    fig, ax = plt.subplots(figsize=(15, 5))
    index_cols = ['Position', 'Guide-Target']

    # target SHAP values
    shap_targets = sequence_pearson_and_shap(data, shap.loc['guide mean'], mode='matches').reset_index()[index_cols + ['SHAP']]
    shap_targets['Position'] += tile_start + 1

    # tile target SHAP values
    shap_target_tiles = pd.DataFrame()
    for tile in range(0, num_tiles):
        shap_targets['Tile'] = tile
        shap_target_tiles = pd.concat([shap_target_tiles, shap_targets])
        shap_targets['Position'] += 1
    shap_target_tiles['Model'] = 'Target (tiles)'

    # average tiled target SHAP values
    num_counts = shap_target_tiles.groupby(index_cols)['SHAP'].count()
    shap_target_mean = shap_target_tiles.groupby(index_cols)['SHAP'].mean()
    shap_target_mean = shap_target_mean.loc[num_counts >= min_overlap, :].reset_index()
    shap_target_mean['Model'] = 'Target (tile mean)'
    shap_target_mean['Tile'] = -1  # dummy value that is unique to support plotting

    # junction SHAP values
    shap.loc['junction mean', 'guide_seq'] = shap.loc['junction mean', 'target_seq'].apply(sequence_complement)
    shap_junctions = sequence_pearson_and_shap(data_junc, shap.loc['junction mean'], mode='matches').reset_index()
    shap_junctions = shap_junctions[index_cols + ['SHAP']]
    shap_junctions['Model'] = 'Junction'
    shap_junctions['Tile'] = -2  # dummy value that is unique to support plotting

    # join data frames
    df = pd.concat([shap_target_tiles, shap_target_mean, shap_junctions])
    df['Target Nucleotide'] = df['Guide-Target'].apply(lambda bp: bp[1])

    # plot the overlay
    sns.lineplot(df, x='Position', y='SHAP', hue='Target Nucleotide', size='Model', style='Model', units='Tile',
                 ax=ax, estimator=None,
                 style_order=['Junction', 'Target (tile mean)', 'Target (tiles)'],
                 sizes={'Target (tiles)': 1.0, 'Target (tile mean)': 1.5, 'Junction': 1.5})

    # manually adjust tiles' alpha value since seaborn doesn't support this
    for line in ax.get_lines()[:8] + ax.get_lines()[10:18] + ax.get_lines()[20:28] + ax.get_lines()[30:38]:
        line.set_alpha(0.3)

    # finalize and save figure
    ax.axvline(shap_target_tiles['Position'].min(), color='black', linestyle='--')
    ax.axvline(shap_target_tiles['Position'].max(), color='black', linestyle='--')
    plt.tight_layout()
    save_fig(fig, fig_path, 'shap-comparison', fig_ext)


def plot_rbp_performance_difference(fig_path: str, fig_ext: str):

    # experiment directory
    exp_dir = os.path.join('experiments', 'junction', 'RBP')

    # loop over models
    for model in ['junction', 'target']:

        # plot performance difference between sequence-only model and one that uses RBP binding predictions
        try:
            performance = pd.DataFrame()
            predictions = pd.DataFrame()
            for config in ['seq_only', 'rbp_junc', 'rbp_nt']:
                index = pd.Index([config.replace('_', ' ')], name='Config.')
                df = pd.read_pickle(os.path.join(exp_dir, model + '-' + config, 'performance.pkl'))
                df.index = index
                performance = pd.concat([performance, df])
                df = pd.read_pickle(os.path.join(exp_dir, model + '-' + config, 'predictions.pkl'))
                df.index = index.repeat(len(df))
                predictions = pd.concat([predictions, df])
            title = model.capitalize() + ' Model'
            fig = plot_performance(performance, predictions, hue='Config.', null='seq only', title=title)
            save_fig(fig, fig_path, 'rbp-performance-' + model, fig_ext)

        except FileNotFoundError:
            continue


def rbp_pearson_and_shap(model: str, rbp_level: str, reduce: bool):

    # try loading SHAP values
    try:
        subdir = model + '-rbp_' + rbp_level + ('_sum_peaks' if reduce else '')
        shap = pd.read_pickle(os.path.join('experiments', 'junction', 'RBP', subdir, 'shap.pkl'))
    except FileNotFoundError:
        return

    # determine context
    # TODO: this only supports symmetric context
    context_5p = context_3p = (len(shap.iloc[0]['target:A']) - len(shap.iloc[0]['target_seq'])) // 2

    # load data
    data = label_and_filter_data(*load_data('junction'))
    data = construct_junctions(data, reduce=(model == 'junction'))
    data, rbp_list = append_rbp_predictions(data, rbp_level, context_5p, context_3p, reduce)
    x, y = np.array(data[rbp_list].values.tolist()), data['target_lfc'].values

    # compute correlations
    y = np.reshape(y, [-1] + [1] * (x.ndim - y.ndim))
    pearson = np.mean((y - y.mean(axis=0)) * (x - x.mean(axis=0)), axis=0) / y.std(axis=0) / x.std(axis=0)
    pearson = pd.DataFrame([np.split(pearson, indices_or_sections=len(rbp_list))], columns=rbp_list)
    if rbp_level == 'junc':
        pearson = pearson.sort_values(by=0, axis=1)
    pearson['value'] = 'Pearson'

    # SHAP values
    shap_mean = np.mean(shap[rbp_list].values.tolist(), axis=0)
    shap_mean = pd.DataFrame([np.split(shap_mean, indices_or_sections=len(rbp_list))], columns=rbp_list)
    shap_mean['value'] = 'SHAP Mean'
    shap_std = np.std(shap[rbp_list].values.tolist(), axis=0)
    shap_std = pd.DataFrame([np.split(shap_std, indices_or_sections=len(rbp_list))], columns=rbp_list)
    shap_std['value'] = 'SHAP Std.'

    # aggregate values
    df = pd.concat([pearson, shap_mean, shap_std])

    # for one-hot peaks, compute conditional average
    data = data[['target_seq'] + rbp_list].set_index('target_seq')
    shap = shap[['target_seq'] + rbp_list].set_index('target_seq')
    data = data.loc[shap.index.values]
    peak = np.array(data[rbp_list].values.tolist())
    shap = np.array(shap[rbp_list].values.tolist())
    if 'nt' in rbp_level and not reduce:
        counts = np.sum(peak, axis=0)
        shap_peak_mean = np.sum(peak * shap, axis=0) / counts
        shap_peak_mean[counts == 0] = 0
        shap_peak_mean = pd.DataFrame([np.split(shap_peak_mean, indices_or_sections=len(rbp_list))], columns=rbp_list)
        shap_peak_mean['value'] = 'SHAP Mean | Peak'
        df = pd.concat([df, shap_peak_mean])
    elif 'nt' in rbp_level and reduce:
        cov = np.mean((peak - np.mean(peak, axis=0)) * (shap - np.mean(shap, axis=0)), axis=0)
        df = pd.concat([df, pd.DataFrame(dict(value=['Covariance'], num_peaks=[cov]))])
        for num_peaks in np.unique(peak):
            counts = np.sum(peak == num_peaks, axis=0)
            shap_peak_mean = np.sum((peak == num_peaks) * shap, axis=0) / counts
            df = pd.concat([df, pd.DataFrame(dict(value=['SHAP Mean | {:d} Peaks'.format(num_peaks)],
                                                  num_peaks=[shap_peak_mean]))])

    return df.set_index('value')


def plot_junction_level_rbp_pearson_and_shap(fig_path: str, fig_ext: str, model: str):

    # get plot values
    df_plot = rbp_pearson_and_shap(model=model, rbp_level='junc', reduce=False)
    df_plot = df_plot.astype(float)

    # plot correlations and SHAP
    fig, ax = plt.subplots(nrows=len(df_plot), figsize=(20, 5 * len(df_plot)))
    fig.suptitle(model.capitalize())
    for i, value in enumerate(df_plot.index):
        ax[i].set_title(model.capitalize() + ' RBP ' + value)
        sns.barplot(df_plot.loc[[value]], ax=ax[i])
        ax[i].set_xticks(np.arange(len(df_plot.columns)))
        ax[i].xaxis.set_ticklabels(df_plot.columns, size=2.5)
        for tick in ax[i].xaxis.get_majorticklabels():
            tick.set_horizontalalignment('right')
            tick.set_rotation(30)
    plt.tight_layout()
    save_fig(fig, fig_path, 'rbp-junction-scores-' + model, fig_ext)


def plot_nt_level_rbp_shap(fig_path: str, fig_ext: str, model: str, rbp_level: str):

    # get plot values
    df_plot = rbp_pearson_and_shap(model=model, rbp_level=rbp_level, reduce=False)

    # SHAP mean vectors
    shap_vectors = np.squeeze(df_plot.loc['SHAP Mean | Peak'].values.tolist())
    i_sort = np.argsort(shap_vectors.sum(axis=1))

    # plot data
    fig, ax = plt.subplots(figsize=tuple(np.flip(shap_vectors.shape) / 10))
    ax.set_title('RBP SHAP values')
    max_abs = np.max(np.abs(shap_vectors))
    sns.heatmap(shap_vectors[i_sort], ax=ax, cmap=sns.color_palette('vlag', as_cmap=True), vmin=-max_abs, vmax=+max_abs)
    ax.set_xticks(np.arange(0, shap_vectors.shape[1] + 1, 5))
    ax.xaxis.set_ticklabels(ax.get_xticks(), size=5)
    ax.set_yticks(np.arange(len(df_plot.columns)))
    ax.yaxis.set_ticklabels(df_plot.columns[i_sort], size=5)
    plt.tight_layout()
    save_fig(fig, fig_path, '-'.join(['rbp', model, rbp_level]), fig_ext)


def plot_nt_level_rbp_peaks(fig_path: str, fig_ext: str, model: str, rbp_level: str):

    # get plot values
    df_plot = rbp_pearson_and_shap(model=model, rbp_level=rbp_level, reduce=True)

    # prepare figure
    shap_vectors = np.squeeze(df_plot.iloc[4:].values.tolist())
    fig, ax = plt.subplots(nrows=2, figsize=tuple(np.flip(shap_vectors.shape) / 10))

    # plot E[SHAP | Num. peaks, Position
    ax[0].set_title('E[SHAP | Num. peaks, Position]')
    max_abs = np.max(np.abs(shap_vectors))
    sns.heatmap(shap_vectors, ax=ax[0], cmap=sns.color_palette('vlag', as_cmap=True), vmin=-max_abs, vmax=+max_abs)
    ax[0].set_ylabel('Num. peaks')
    ax[0].set_xlabel('Position')

    # plot Cov[SHAP, Num. peaks | Position]
    ax[1].set_title('Cov[SHAP, Num. peaks | Position]')
    shap_vectors = np.array(df_plot.loc['Covariance'].values.tolist())[0]
    max_abs = np.max(np.abs(shap_vectors))
    sns.heatmap(shap_vectors, ax=ax[1], cmap=sns.color_palette('vlag', as_cmap=True), vmin=-max_abs, vmax=+max_abs)
    ax[1].set_xlabel('Position')
    ax[1].yaxis.set_ticklabels(['Covariance'], size=5)

    # finalize figure
    plt.tight_layout()
    plt.show()
    save_fig(fig, fig_path, '-'.join(['num_peaks', model, rbp_level]), fig_ext)


if __name__ == '__main__':

    # ensure text is text in images
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['svg.fonttype'] = 'none'

    # custom junction figure directory
    figure_path = os.path.join('figures', 'junction', 'custom')
    figure_ext = '.pdf'

    # dataset differences plot
    plot_training_set_differences(figure_path, figure_ext)

    # target vs junction SHAP comparison
    plot_target_vs_junction_shap(figure_path, figure_ext)

    # RBP performance differences
    plot_rbp_performance_difference(figure_path, figure_ext)

    # # junction-level RBP importance scores
    # plot_junction_level_rbp_pearson_and_shap(figure_path, figure_ext, model='junction')
    # plot_junction_level_rbp_pearson_and_shap(figure_path, figure_ext, model='target')
    #
    # # nucleotide-level RBP importance scores
    # for mode in itertools.product(['junction', 'target'], ['nt', 'nt_relaxed']):
    #     plot_nt_level_rbp_shap(figure_path, figure_ext, *mode)
    #     plot_nt_level_rbp_peaks(figure_path, figure_ext, *mode)

    plt.show()
