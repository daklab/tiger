import os
import utils
import numpy as np
import pandas as pd
import seaborn as sns
from analysis import sequence_pearson_and_shap, save_fig
from data import load_data, label_and_filter_data, LFC_COLS
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

TIGER_EXON = 'TIGER (Wessels et al. 2023)'
TIGER_JUNC = 'TIGER trained on our junction screen'
TIGER_BASS = 'TIGER BASS (this paper)'
CHENG = 'DeepCas13 (Cheng et al. 2023)'
WEI = 'Wei et al. 2023'


def plot_exon_vs_junc_shap_and_pearson(fig_path: str, fig_ext: str):

    # load SHAP values
    try:
        exon_shap = pd.read_pickle(os.path.join('experiments', 'off-target', 'SHAP', 'pm', 'targets', 'shap.pkl'))
        junc_shap = pd.read_pickle(os.path.join('experiments', 'junction', 'SHAP', 'pm', 'targets', 'shap.pkl'))
    except FileNotFoundError:
        return None

    # load datasets
    exon_data = load_data(dataset='off-target', pm_only=True)[0]
    exon_data['observed_lfc'] = exon_data[LFC_COLS].mean(axis=1)
    exon_data = exon_data.loc[exon_data['guide_seq'].isin(exon_shap['guide_seq'])]
    junc_data = load_data(dataset='junction')[0]
    junc_data['observed_lfc'] = junc_data[LFC_COLS].mean(axis=1)
    junc_data = junc_data.loc[junc_data['guide_seq'].isin(junc_shap['guide_seq'])]

    # get pearson and shap values
    df_exon = sequence_pearson_and_shap(exon_data, exon_shap, mode='matches').reset_index()
    df_exon['Type'] = 'exon'
    df_junc = sequence_pearson_and_shap(junc_data, junc_shap, mode='matches').reset_index()
    df_junc['Type'] = 'junction'
    df = pd.concat([df_exon, df_junc])

    # plot results
    for value in ['Pearson', 'SHAP']:

        # combined plot
        fig, ax = plt.subplots()
        sns.lineplot(df, x='Position', y='SHAP', hue='Guide-Target', style='Type', ax=ax)

        # facet plot
        g = sns.FacetGrid(df, col='Guide-Target', hue='Guide-Target')
        g.map_dataframe(sns.lineplot, x='Position', y=value, style='Type')
        g.axes.flat[-1].legend(*ax.get_legend_handles_labels(), loc='center right', bbox_to_anchor=(1.75, 0.5))
        plt.tight_layout()

        # close figure we don't want
        plt.close(fig)

        # save the figure we do want
        save_fig(g.figure, fig_path, 'exon-junc-' + value.lower(), fig_ext)


def plot_target_vs_junction_shap(fig_path: str, fig_ext: str, min_overlap: int = 3):
    assert min_overlap > 0

    # load SHAP values
    try:
        sub_dir = os.path.join('SHAP', 'pm', 'targets', 'shap.pkl')
        target_shap = pd.read_pickle(os.path.join('experiments', 'junction', sub_dir))
        junction_shap = pd.read_pickle(os.path.join('experiments', 'junction-splice-sites', sub_dir))
    except FileNotFoundError:
        return

    # load datasets
    target_data = load_data(dataset='junction')[0]
    target_data['observed_lfc'] = target_data[LFC_COLS].mean(axis=1)
    target_data = target_data.loc[target_data['guide_seq'].isin(target_shap['guide_seq'])]
    junction_data = load_data(dataset='junction-splice-sites')[0]
    junction_data['observed_lfc'] = junction_data[LFC_COLS].mean(axis=1)
    junction_data = junction_data.loc[junction_data['guide_seq'].isin(junction_shap['guide_seq'])]

    # shapes
    guide_len = target_data['guide_seq'].apply(len).unique()[0]
    num_tiles = target_data['junction_olap_5p'].nunique()
    tile_start = junction_data['target_seq'].apply(len).unique()[0] // 2
    tile_start += int(target_data['junction_olap_5p'].min() * guide_len)

    # average SHAP values
    shap_targets = sequence_pearson_and_shap(target_data, target_shap, mode='matches').reset_index()
    shap_junctions = sequence_pearson_and_shap(junction_data, junction_shap, mode='matches').reset_index()

    # prepare SHAP figure
    fig, ax = plt.subplots(figsize=(15, 5))
    index_cols = ['Position', 'Guide-Target']

    # tile target SHAP values
    shap_targets['Position'] += tile_start
    shap_target_tiles = pd.DataFrame()
    for tile in range(0, num_tiles):
        shap_targets['Tile'] = tile
        shap_target_tiles = pd.concat([shap_target_tiles, shap_targets])
        shap_targets['Position'] += 1
    shap_target_tiles['SHAP Source'] = 'Guide model (tiles)'

    # average tiled target SHAP values
    num_counts = shap_target_tiles.groupby(index_cols)['SHAP'].count()
    shap_target_mean = shap_target_tiles.groupby(index_cols)['SHAP'].mean()
    shap_target_mean = shap_target_mean.loc[num_counts >= min_overlap, :].reset_index()
    shap_target_mean['SHAP Source'] = 'Guide model (tile mean)'
    shap_target_mean['Tile'] = -1  # dummy value that is unique to support plotting

    # junction SHAP values
    shap_junctions['SHAP Source'] = 'Junction model'
    shap_junctions['Tile'] = -2  # dummy value that is unique to support plotting

    # join data frames
    df = pd.concat([shap_target_tiles, shap_target_mean, shap_junctions])
    df['Target Nucleotide'] = df['Guide-Target'].apply(lambda bp: bp[1])

    # plot the overlay
    sns.lineplot(df, x='Position', y='SHAP',
                 hue='Target Nucleotide', size='SHAP Source', style='SHAP Source', units='Tile',
                 ax=ax, estimator=None,
                 style_order=['Junction model', 'Guide model (tile mean)', 'Guide model (tiles)'],
                 sizes={'Guide model (tiles)': 1.0, 'Guide model (tile mean)': 1.5, 'Junction model': 1.5})

    # manually adjust tiles' alpha value since seaborn doesn't support this
    for line in ax.get_lines()[:8] + ax.get_lines()[10:18] + ax.get_lines()[20:28] + ax.get_lines()[30:38]:
        line.set_alpha(0.3)

    # finalize and save figure
    ax.set_xlim([1, 100])
    ax.set_xticks([1.5, 25.5, 50.5, 75.5, 100.5], ['-50', '-25', '0', '+25', '+50'])
    ax.axvline(shap_target_tiles['Position'].min(), color='black', linestyle='--')
    ax.axvline(shap_target_tiles['Position'].max(), color='black', linestyle='--')
    plt.tight_layout()
    save_fig(fig, fig_path, 'shap-comparison', fig_ext)


def load_gene_essentiality():
    # compute gene essentiality
    data = label_and_filter_data(*load_data('junction'), method='NoFilter')
    gene_essentiality = data.groupby('gene')['observed_label'].mean().rename('essentiality')
    return gene_essentiality


def load_predictions(dataset: str):

    # load dataset
    exon_path = os.path.join('predictions', 'junction', 'tiger', 'no_indels')
    junc_path = os.path.join('predictions', 'junction', 'tiger-junc', 'pm')
    bass_path = os.path.join('predictions', 'junction', 'tiger-bass', 'pm')
    if dataset == 'junction':
        csv_file = 'predictions_junc.csv'
    elif dataset == 'junction-qpcr':
        csv_file = 'predictions_qpcr.csv'
    else:
        raise NotImplementedError
    pred_tiger_exon = os.path.join(exon_path, csv_file)
    pred_tiger_junc = os.path.join(junc_path, csv_file)
    pred_tiger_bass = os.path.join(bass_path, csv_file)
    if os.path.exists(pred_tiger_exon) and os.path.exists(pred_tiger_junc) and os.path.exists(pred_tiger_bass):
        pred_tiger_exon = pd.read_csv(pred_tiger_exon)
        pred_tiger_junc = pd.read_csv(pred_tiger_junc)
        pred_tiger_bass = pd.read_csv(pred_tiger_bass)
    else:
        return None

    # force utilization of normalized predictions
    for pred in [pred_tiger_junc, pred_tiger_bass]:
        if 'predicted_lfc_normalized' in pred.columns:
            del pred['predicted_lfc']
            pred.rename(columns={'predicted_lfc_normalized': 'predicted_lfc'}, inplace=True)

    # load other model predictions
    pred_cheng = pd.read_csv(os.path.join('predictions (other models)', 'DeepCas13', dataset, 'predictions.csv'))
    merge_columns = ['guide_seq'] + list(set(pred_tiger_exon.columns) - set(pred_cheng.columns))
    pred_cheng = pd.merge(pred_cheng, pred_tiger_exon[merge_columns])
    pred_wei = pd.read_csv(os.path.join('predictions (other models)', 'Konermann', dataset, 'predictions.csv'))
    merge_columns = ['guide_seq'] + list(set(pred_tiger_exon.columns) - set(pred_wei.columns))
    pred_wei = pd.merge(pred_wei, pred_tiger_exon[merge_columns])
    pred_wei = pd.merge(pred_wei, pred_tiger_exon[['guide_seq', 'gene', 'observed_lfc']])

    # model indices
    index_tiger_exon = pd.Index(data=[TIGER_EXON], name='Model')
    index_tiger_junc = pd.Index(data=[TIGER_JUNC], name='Model')
    index_tiger_bass = pd.Index(data=[TIGER_BASS], name='Model')
    index_cheng = pd.Index(data=[CHENG], name='Model')
    index_wei = pd.Index(data=[WEI], name='Model')

    # concatenate predictions
    predictions = pd.concat([
        pred_tiger_exon.set_index(index_tiger_exon.repeat(len(pred_tiger_exon))),
        pred_tiger_junc.set_index(index_tiger_junc.repeat(len(pred_tiger_junc))),
        pred_tiger_bass.set_index(index_tiger_bass.repeat(len(pred_tiger_bass))),
        pred_cheng.set_index(index_cheng.repeat(len(pred_cheng))),
        pred_wei.set_index(index_wei.repeat(len(pred_wei))),
    ])

    return predictions


def add_sea_bass_slopes(predictions):
    # ensure every prediction has both observed LFC and observed slope
    tiger_junc_observations = predictions.loc[TIGER_JUNC, ['guide_seq', 'observed_lfc']]
    tiger_junc_observations.rename(columns={'observed_lfc': 'Day 21 LFC'}, inplace=True)
    tiger_bass_observations = predictions.loc[TIGER_BASS, ['guide_seq', 'observed_lfc']]
    tiger_bass_observations = tiger_bass_observations.rename(columns={'observed_lfc': 'Sea-bass Slope'})
    observations = pd.merge(tiger_junc_observations, tiger_bass_observations, on='guide_seq').set_index('guide_seq')
    return pd.merge(predictions, observations, how='left', left_on='guide_seq', right_index=True)


def sea_bass_benefit(fig_path: str, fig_ext: str, gene_threshold: float, p_sig: float):
    # load predictions
    predictions = load_predictions(dataset='junction')
    if predictions is None:
        return
    predictions = predictions.loc[predictions.index.isin([TIGER_EXON, TIGER_JUNC, TIGER_BASS])]
    predictions = predictions.loc[predictions['guide_seq'].isin(predictions.loc[TIGER_BASS, 'guide_seq'])]

    # filter high-confidence essential genes
    gene_essentiality = load_gene_essentiality()
    essential_genes = gene_essentiality[gene_essentiality >= gene_threshold].reset_index()['gene']
    essential_genes.to_csv(os.path.join(fig_path, 'essential_genes-{:.2f}.csv'.format(gene_threshold)), index=False)
    predictions = predictions.loc[predictions['gene'].isin(essential_genes)]

    # ensure SEA BASS slopes and Day 21 LFC are available to all models
    predictions = add_sea_bass_slopes(predictions)
    del predictions['observed_label']

    # measure performances
    performance = pd.DataFrame()
    for model in predictions.index.unique():
        for observation in ['Day 21 LFC', 'Sea-bass Slope']:
            df = predictions.loc[model, ['predicted_lfc', 'predicted_score', observation]]
            performance = pd.concat([performance, pd.DataFrame(index=pd.Index([model], name='Model'), data={
                'Observation': [observation],
                'Pearson': [pearsonr(df['predicted_lfc'], df[observation])[0]],
                # 'Spearman (Score)': [-spearmanr(df['predicted_score'], df[observation])[0]],
            })])

    # run statistical tests
    performance = pd.concat([utils.statistical_tests(
        reference_model=TIGER_BASS,
        performance=performance.loc[performance['Observation'] == observation].copy(),
        predictions=predictions,
    ) for observation in ['Day 21 LFC', 'Sea-bass Slope']])
    performance.to_csv(os.path.join(fig_path, 'sea-bass-benefit-{:.2f}.csv'.format(gene_threshold)))

    # plot results
    fig, ax = plt.subplots()
    sns.barplot(performance.reset_index(), x='Observation', y='Pearson', hue='Model')
    for i, model in enumerate(performance.index.unique()):
        for j, observation in enumerate(performance['Observation'].unique()):
            x = ax.containers[i][j].get_x() + ax.containers[i][j].get_width() / 2
            y = ax.containers[i].datavalues[j]
            if model == TIGER_BASS:
                ax.text(x, y + 0.01, '$H_0$', horizontalalignment='center')
            elif performance.loc[performance['Observation'] == observation, 'Pearson log10(p)'].loc[model] < p_sig:
                ax.text(x, y + 0.01, '*', horizontalalignment='center')
    ax.set_ylabel('Pearson: Predicted LFC vs Observation')
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    save_fig(fig, fig_path, 'sea-bass-benefit-{:.2f}'.format(gene_threshold), fig_ext)


def gene_filtering_effect_all_models(fig_path: str, fig_ext: str):

    # load predictions and join sea-bass slope estimates
    predictions = load_predictions(dataset='junction')
    if predictions is None:
        return
    predictions = predictions.loc[predictions.index.isin([TIGER_EXON, TIGER_JUNC, TIGER_BASS])]

    # keep only common set of guides
    guides = set.intersection(*[set(predictions.loc[idx, 'guide_seq'].unique()) for idx in predictions.index.unique()])
    predictions = predictions.loc[predictions['guide_seq'].isin(guides)].copy()

    # loop over essentiality filtering values
    performance = pd.DataFrame()
    gene_essentiality = load_gene_essentiality()
    for essentiality in np.arange(0.05, 0.50, 0.05):
        genes = list(gene_essentiality.index[gene_essentiality >= essentiality])
        pred_filtered = predictions.loc[predictions['gene'].isin(genes)]
        for idx in predictions.index.unique():
            df = pred_filtered.loc[idx]
            _, _, auprc, _ = utils.classification_metrics(df['observed_label'], df['predicted_score'])
            performance = pd.concat([performance, pd.DataFrame(index=pd.Index([idx], name='Model'), data={
                'AUPRC': auprc,
                'Essentiality': essentiality
            })])
    performance.reset_index(inplace=True)

    # plot sea-bass effect
    fig, ax = plt.subplots()
    sns.lineplot(performance, x='Essentiality', y='AUPRC', hue='Model', ax=ax, alpha=0.5)
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    save_fig(fig, fig_path, 'gene-filtering-all-auprc', fig_ext)


def gene_filtering_effect_tiger_bass(fig_path: str, fig_ext: str):
    # load TIGER-BASS predictions with both observed LFC and slopes
    predictions = load_predictions(dataset='junction')
    if predictions is None:
        return
    predictions = add_sea_bass_slopes(predictions).loc[TIGER_BASS]

    # compute gene essentiality
    gene_essentiality = predictions.groupby('gene')['observed_label'].mean()

    # loop over essentiality filtering values
    performance = pd.DataFrame()
    for essentiality in np.arange(0.05, 0.45, 0.05):
        genes = list(gene_essentiality.index[gene_essentiality >= essentiality])
        pred_filtered = predictions.loc[predictions['gene'].isin(genes)]
        index = pd.Index([essentiality], name='Essentiality')
        df = utils.measure_performance(pred_filtered, obs_var='Sea-bass Slope', index=index, silence=True)
        df['Guide Retention Ratio'] = len(pred_filtered) / len(predictions)
        df['Gene Retention Ratio'] = len(genes) / len(gene_essentiality)
        performance = pd.concat([performance, df])
    performance.reset_index(inplace=True)

    # melt plot
    df = performance.melt(
        id_vars='Essentiality',
        value_vars=['Guide Retention Ratio', 'Gene Retention Ratio', 'Pearson', 'Spearman', 'AUROC', 'AUPRC'],
        var_name='Metric',
        value_name='Value')

    # plot performance
    g = sns.FacetGrid(df, col='Metric', col_wrap=2, sharey=False)
    g.map(sns.lineplot, 'Essentiality', 'Value', 'Metric')
    save_fig(g.figure, fig_path, 'gene-filtering-tiger-bass-summary', fig_ext)

    # pull out ROC/PRC vectors
    df = performance.set_index('Essentiality')[['ROC', 'PRC']]
    for col, key in [('ROC', 'fpr'), ('ROC', 'tpr'), ('PRC', 'precision'), ('PRC', 'recall')]:
        df[key] = df[col].apply(lambda d: d[key])

    # plot filtering effect on PRC
    fig, ax = plt.subplots()
    norm = plt.Normalize(df.index.min(), df.index.max())
    cm = plt.cm.rainbow
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    for ess in df.index.unique():
        ax.plot(df.loc[ess, 'recall'], df.loc[ess, 'precision'], color=cm(norm(ess)))
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.colorbar(sm)
    save_fig(fig, fig_path, 'gene-filtering-tiger-bass-prc', fig_ext)


def qpcr_comparison(fig_path: str, fig_ext: str):

    # load predictions
    predictions = load_predictions(dataset='junction-qpcr')
    if predictions is None:
        return

    # qPCR scatter plot
    rename_dict = {'predicted_score': 'Model Predictions', 'observed_lfc': 'qPCR: RNA KD Rel. to Control'}
    x, y = tuple(rename_dict.values())
    df = predictions.reset_index().rename(columns={**rename_dict, **dict(gene='Gene')})
    g = sns.lmplot(df, x=x, y=y, col='Model', scatter=False)
    for mdl, ax in g.axes_dict.items():
        sns.scatterplot(df.loc[df.Model == mdl], x=x, y=y, hue='Gene', ax=ax)
        r, p = pearsonr(df.loc[df.Model == mdl, x], df.loc[df.Model == mdl, y])
        ax.text(0.35, 1.05, 'R = {:.2f}'.format(r) + (', p < 0.001' if p < 0.001 else ''))
    save_fig(g.figure, fig_path, 'qpcr', fig_ext)


if __name__ == '__main__':

    # ensure text is text in images
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['svg.fonttype'] = 'none'

    # custom junction figure directory
    figure_path = os.path.join('figures', 'junction', 'custom')
    os.makedirs(figure_path, exist_ok=True)
    figure_ext = '.pdf'

    # exon vs junction sequence feature comparison
    plot_exon_vs_junc_shap_and_pearson(figure_path, figure_ext)

    # target vs junction SHAP comparison
    plot_target_vs_junction_shap(figure_path, figure_ext, min_overlap=4)

    # plot sea-bass benefit
    sea_bass_benefit(figure_path, figure_ext, gene_threshold=0.25, p_sig=0.0001)

    # essential gene filter vs test performance
    gene_filtering_effect_all_models(figure_path, figure_ext)
    gene_filtering_effect_tiger_bass(figure_path, figure_ext)

    # plot model comparison
    qpcr_comparison(figure_path, figure_ext)

    plt.show()
