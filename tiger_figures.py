import os
import utils
import numpy as np
import pandas as pd
import seaborn as sns
from analysis import plot_performance, plot_performance_by_type, save_fig
from data import load_data, label_and_filter_data
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def replicate_performance(data):

    # using a held-out replicate predict mean({replicates} \ held-out replicate)
    predictions = pd.DataFrame()
    for replicate in ['lfc_r1', 'lfc_r2', 'lfc_r3']:
        target_lfc = data[list({'lfc_r1', 'lfc_r2', 'lfc_r3'} - {replicate})].mean(axis=1)
        target_label = data['target_label']
        predicted_lfc = data[replicate]
        keep = ~target_lfc.isna() & ~predicted_lfc.isna()
        predictions = pd.concat([predictions, pd.DataFrame({
            'target_lfc': target_lfc[keep],
            'target_label': target_label[keep],
            'predicted_lfc': predicted_lfc[keep]
        })])

    # compute metrics
    performance = utils.measure_performance(predictions,
                                            index=pd.Index(data=['Replicates'], name='Model'),
                                            silence=True)

    return performance


def others_pm_performance(test_set):

    def cheng_lfc_transform(lfc):
        b = np.log((1 / 0.3) - 1)
        n = (np.log((1 / 0.7) - 1) - b) / 0.3
        x = n * lfc - b
        return (1 + np.exp(-x)) ** -1

    # directory with other model's predictions
    base_dir = 'predictions (other models)'

    # relevant columns
    index_cols = ['gene', 'guide_seq']
    lfc_cols = ['target_lfc', 'target_label', 'predicted_lfc']

    # load dataset and make sure our forthcoming assumptions hold
    data = load_data(test_set, pm_only=True)
    data = label_and_filter_data(*data, nt_quantile=0.01, method='NoFilter')
    data = data[['gene', 'guide_id', 'guide_seq', 'target_lfc', 'target_label']]
    assert len(data) == data['guide_seq'].nunique()

    # Wessels predictions
    if test_set == 'off-target':
        pred_wessels = pd.read_csv(os.path.join(base_dir, 'RandomForest', test_set, 'OFF-target-Lib_L1OVC_RFnbt.csv'))
        pred_wessels['guide_id'] = pred_wessels['Guide_Name'].apply(lambda s: s.split('_')[1])
        del pred_wessels['Guide_Name']
        pred_wessels.rename(columns={
            'Gene_Holdout': 'gene',
            'normCS.D30': 'target_lfc',
            'predicted': 'predicted_lfc'}, inplace=True)
        pred_wessels.set_index(['gene', 'guide_id'], inplace=True)
        pred_wessels = pred_wessels.join(data.set_index(['gene', 'guide_id'])[['guide_seq', 'target_label']])
        pred_wessels = pred_wessels.reset_index()[index_cols + lfc_cols]
    else:
        pred_wessels = pd.DataFrame()
        for gene_csv in os.listdir(os.path.join(base_dir, 'RandomForest', test_set)):
            df = pd.read_csv(os.path.join(base_dir, 'RandomForest', test_set, gene_csv))
            df['guide_seq'] = df['GuideSeq'].apply(lambda seq: seq[::-1])
            df['predicted_lfc'] = -df['GuideScores']
            df = df[['guide_seq', 'predicted_lfc']]
            df = df.merge(data, on='guide_seq')[index_cols + lfc_cols]
            pred_wessels = pd.concat([pred_wessels, df])

    # Cheng et al. and Wei et al. predictions
    pred_cheng = pd.DataFrame()
    pred_wei = pd.DataFrame()
    for gene in data['gene'].unique():

        # retrieve and aggregate Cheng et al. predictions for this gene
        df = pd.read_csv(os.path.join(base_dir, 'DeepCas13', test_set, gene + '.csv'))
        df['gene'] = gene
        df['guide_seq'] = df['sgRNA Sequence'].apply(lambda seq: seq[::-1])
        df['predicted_lfc'] = df['Score'].apply(cheng_lfc_transform)
        df = df.merge(data, on=index_cols)[index_cols + lfc_cols]
        assert abs(len(df) - len(data[data.gene == gene])) <= 6
        pred_cheng = pd.concat([pred_cheng, df])

        # retrieve and aggregate Wei et al. predictions for this gene
        df = pd.read_csv(os.path.join(base_dir, 'Konermann', test_set, gene + '.csv'))
        df['gene'] = gene
        df['guide_seq'] = df['guide'].apply(lambda seq: seq[:23][::-1])
        df['predicted_lfc'] = -df['predicted_value_sigmoid']
        df = df.merge(data, on=index_cols)[index_cols + lfc_cols]
        assert abs(len(df) - len(data[data.gene == gene])) <= 6
        pred_wei = pd.concat([pred_wei, df])

    # model indices
    index_wessels = pd.Index(data=['Wessels'], name='Model')
    index_cheng = pd.Index(data=['Cheng'], name='Model')
    index_wei = pd.Index(data=['Wei'], name='Model')

    # concatenate performances
    performance = pd.concat([
        utils.measure_performance(pred_wessels, index_wessels, silence=True),
        utils.measure_performance(pred_cheng, index_cheng, silence=True),
        utils.measure_performance(pred_wei, index_wei, silence=True),
    ])

    # concatenate predictions
    predictions = pd.concat([
        pred_wessels.set_index(index_wessels.repeat(len(pred_wessels))),
        pred_cheng.set_index(index_cheng.repeat(len(pred_cheng))),
        pred_wei.set_index(index_wei.repeat(len(pred_wei))),
    ])

    return performance, predictions


def on_target_performance(holdout):

    # load our results
    try:
        # combined model
        dir_combined = os.path.join('predictions', 'off-target', 'no_indels', holdout)
        pred_combined = pd.read_pickle(os.path.join(dir_combined, 'predictions.pkl'))
        pred_combined = pred_combined[pred_combined.guide_type == 'PM']
        perf_combined = utils.measure_performance(pred_combined, silence=True)
        pred_combined['Model'] = perf_combined['Model'] = 'Ours (combined)'

        # on-target performance
        dir_on_target = os.path.join('predictions', 'off-target', 'pm', holdout)
        pred_on_target = pd.read_pickle(os.path.join(dir_on_target, 'predictions.pkl'))
        perf_on_target = pd.read_pickle(os.path.join(dir_on_target, 'performance.pkl'))
        pred_on_target['Model'] = perf_on_target['Model'] = 'Ours (on-target)'

    except FileNotFoundError:
        return None

    # aggregate performance and predictions
    performance = pd.concat([perf_combined, perf_on_target]).set_index('Model')
    predictions = pd.concat([pred_combined, pred_on_target]).set_index('Model')

    # replicate performance
    dataset = holdout if holdout == 'flow-cytometry' else 'off-target'
    data = label_and_filter_data(*load_data(dataset, pm_only=True), nt_quantile=0.01, method='NoFilter')
    performance_replicates = replicate_performance(data)

    # append other model's performance and predictions
    test_set = 'flow-cytometry' if holdout == 'flow-cytometry' else 'off-target'
    performance_others, predictions_others = others_pm_performance(test_set)
    performance = pd.concat([performance, performance_others, performance_replicates])
    predictions = pd.concat([predictions, predictions_others])

    # plot performance
    title = 'On-target Performance: ' + holdout
    fig = plot_performance(performance, predictions, hue='Model', null='Ours (combined)', title=title)
    save_fig(fig, os.path.join('figures', 'tiger'), 'on-target-' + holdout, '.svg')


def off_target_performance(holdout):

    # load results
    try:
        # combined model
        dir_combined = os.path.join('predictions', 'off-target', 'no_indels', holdout)
        pred_combined = pd.read_pickle(os.path.join(dir_combined, 'predictions.pkl'))
        pred_combined['Model'] = 'Ours (combined)'

        # off-target performance
        dir_off_target = os.path.join('predictions', 'off-target', 'mm', holdout)
        pred_off_target = pd.read_pickle(os.path.join(dir_off_target, 'predictions.pkl'))
        pred_off_target['Model'] = 'Ours (off-target)'

    except FileNotFoundError:
        return None

    # guide type performance
    predictions = pd.concat([pred_combined, pred_off_target]).set_index('Model')
    performance = utils.measure_guide_type_performance(predictions, reference='Ours (combined)')

    # plot performance
    title = 'Off-target Performance: ' + holdout
    fig = plot_performance_by_type(performance[performance.guide_type != 'PM'], 'Ours (combined)', title)
    plt.tight_layout()
    save_fig(fig, os.path.join('figures', 'tiger'), 'off-target-' + holdout, '.svg')


def delta_pearson_performance(holdout):
    if holdout == 'guides':
        return

    # load results
    try:
        # combined model
        predictions = os.path.join('predictions', 'off-target', 'no_indels', holdout)
        predictions = pd.read_pickle(os.path.join(predictions, 'predictions.pkl'))
        predictions['Model'] = 'Ours (combined)'
        predictions.set_index('Model', inplace=True)

    except FileNotFoundError:
        return None

    # guide type performance
    performance = utils.measure_guide_type_performance(predictions)

    # guide-type delta performance
    predictions['target_lfc'] -= predictions['target_pm_lfc']
    predictions['predicted_lfc'] -= predictions['predicted_pm_lfc']
    delta_performance = utils.measure_guide_type_performance(predictions[predictions.guide_type != 'PM'])

    # plot absolute vs delta performance
    performance['Prediction'] = 'absolute'
    delta_performance['Prediction'] = 'delta'
    performance = pd.concat([performance.loc['Ours (combined)'], delta_performance])
    performance.rename(columns={'guide_type': 'Guide type'}, inplace=True)
    order = ['PM', 'SM', 'DM', 'RDM', 'TM', 'RTM', 'SI', 'CI', 'DI', 'SD', 'CD', 'DD']
    [order.remove(gt) for gt in set(order) - set(performance['Guide type'].unique())]
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(data=performance, x='Guide type', y='Pearson', hue='Prediction', order=order, ax=ax)
    ax.set_title('Delta Pearson: ' + holdout)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    save_fig(fig, os.path.join('figures', 'tiger'), 'delta-' + holdout, '.svg')


def compute_titration_ratios(df_tap):

    # compute titration ratios
    df_tap['Observed ratio'] = 2 ** -(df_tap['target_lfc_normalized'] - df_tap['target_pm_lfc_normalized'])
    df_tap['Predicted ratio'] = 2 ** -(df_tap['predicted_lfc_normalized'] - df_tap['predicted_pm_lfc_normalized'])

    # index for SM guides with active targets
    active_targets = df_tap.loc[(df_tap.guide_type == 'PM') & (df_tap.target_label == 1), 'target_seq']
    idx = df_tap.target_seq.isin(active_targets) & df_tap.guide_type.isin({'SM'})

    # scale ratios to [0, 1] for SM guides with active targets
    scale = df_tap.loc[idx, 'Predicted ratio'].quantile(0.01)
    df_tap['Observed ratio'] = (df_tap['Observed ratio'] - scale) / (1 - scale)
    df_tap['Predicted ratio'] = (df_tap['Predicted ratio'] - scale) / (1 - scale)

    # bias correction fit with SM guides with active targets
    p = np.polyfit(x=df_tap.loc[idx, 'Predicted ratio'], y=df_tap.loc[idx, 'Observed ratio'], deg=1)
    df_tap['Predicted ratio'] = df_tap['Predicted ratio'].apply(lambda x: np.polyval(p, x))

    return df_tap


def titration_confusion_matrix(df_tap, title):

    # keep only guides with active target parents
    active_targets = df_tap.loc[(df_tap.guide_type == 'PM') & (df_tap.target_label == 1), 'target_seq']
    df_tap = df_tap[df_tap.target_seq.isin(active_targets) & (df_tap.guide_type.isin({'SM'}))]

    # place observed and predicted ratios into bins
    bins = np.arange(0.2, 1.0, .2)
    df_tap['Observed bin'] = np.digitize(df_tap['Observed ratio'], bins)
    df_tap['Predicted bin'] = np.digitize(df_tap['Predicted ratio'], bins)

    # compute confusion matrix with each column normalized
    mtx = confusion_matrix(df_tap['Observed bin'], df_tap['Predicted bin'], normalize='pred')

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.suptitle(title)
    x_ticks = ['Pred. < {:.2f}'.format(bins[0])]
    y_ticks = ['Obs. < {:.2f}'.format(bins[0])]
    for i, lower in enumerate(bins[:-1]):
        interval = '[{:.2f}, {:.2f})'.format(lower, bins[i+1])
        x_ticks += [interval]
        y_ticks += [interval]
    x_ticks += ['Pred. $\\geq$ {:.2f}'.format(bins[-1])]
    y_ticks += ['Obs. $\\geq$ {:.2f}'.format(bins[-1])]
    color_map = sns.color_palette("light:#DE2D26", as_cmap=True)
    sns.heatmap(mtx, annot=True, fmt='.2f', cmap=color_map, xticklabels=x_ticks, yticklabels=y_ticks, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    return fig


def titration_performance():

    # load results
    try:
        # HEK293 training data
        path_hek = os.path.join('predictions', 'off-target', 'no_indels', 'targets')
        predictions_hek = pd.read_pickle(os.path.join(path_hek, 'predictions.pkl'))
        predictions_hek['cell'] = 'HEK293'

        # HAP1 validation data
        path_hap = os.path.join('predictions', 'off-target', 'no_indels', 'hap-titration')
        predictions_hap = pd.read_pickle(os.path.join(path_hap, 'predictions.pkl'))
        predictions_hap['cell'] = 'HAP1'
        df_uid = load_data('hap-titration')[0][['guide_id', 'guide_seq', 'target_seq']]
        predictions_hap = predictions_hap.merge(df_uid, on=['guide_seq', 'target_seq'])

    except FileNotFoundError:
        return None

    # compute titration ratios
    predictions_hek = compute_titration_ratios(predictions_hek)
    predictions_hap = compute_titration_ratios(predictions_hap)

    # save predictions
    predictions_hek.to_csv(os.path.join(path_hek, 'titration_hek.csv'), index=False)
    predictions_hap.to_csv(os.path.join(path_hap, 'titration_hap.csv'), index=False)

    # plot titration performance
    fig_hek = titration_confusion_matrix(predictions_hek, title='SM Titration: HEK293')
    save_fig(fig_hek, os.path.join('figures', 'tiger'), 'titration_hek', '.svg')
    fig_hap = titration_confusion_matrix(predictions_hap, title='SM Titration: HAP1')
    save_fig(fig_hap, os.path.join('figures', 'tiger'), 'titration_hap', '.svg')


def plot_training_set_differences():

    # load relevant files
    predictions_train_flow_dir = os.path.join('predictions', 'flow-cytometry', 'no_indels', 'off-target')
    predictions_train_hek_dir = os.path.join('predictions', 'off-target', 'no_indels', 'genes')
    try:
        predictions_train_flow = pd.read_pickle(os.path.join(predictions_train_flow_dir, 'predictions.pkl'))
        performance_train_flow = pd.read_pickle(os.path.join(predictions_train_flow_dir, 'performance.pkl'))
        predictions_train_hek = pd.read_pickle(os.path.join(predictions_train_hek_dir, 'predictions.pkl'))
        performance_train_hek = pd.read_pickle(os.path.join(predictions_train_hek_dir, 'performance.pkl'))
    except FileNotFoundError:
        return

    # set indices
    predictions_train_flow['Training'] = 'flow cytometry'
    performance_train_flow['Training'] = 'flow cytometry'
    predictions_train_hek['Training'] = 'this screen'
    performance_train_hek['Training'] = 'this screen'

    # combine predictions and performance
    predictions = pd.concat([predictions_train_flow.set_index('Training'),
                             predictions_train_hek.set_index('Training')])
    performance = pd.concat([performance_train_flow.set_index('Training'),
                             performance_train_hek.set_index('Training')])

    # plot performance
    fig = plot_performance(performance, predictions, hue='Training', null='this screen', title='Dataset differences')
    save_fig(fig, os.path.join('figures', 'tiger', 'no_indels'), 'dataset_differences', '.svg')


if __name__ == '__main__':

    # ensure text is text in images
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['svg.fonttype'] = 'none'

    # on-target and off-target performance plots
    for validation_strategy in ['genes', 'guides', 'targets', 'flow-cytometry']:
        on_target_performance(holdout=validation_strategy)
        off_target_performance(holdout=validation_strategy)
        delta_pearson_performance(holdout=validation_strategy)

    # titration performance plots
    titration_performance()

    # dataset differences plot
    plot_training_set_differences()

    plt.show()
