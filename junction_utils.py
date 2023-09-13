import os
import numpy as np
import pandas as pd


def construct_junctions(data, reduce):
    # convert to junction data
    data['junction_seq'] = data['5p_context'] + data['target_seq'] + data['3p_context']
    data_junction_group_by = data[['junction_seq', 'observed_lfc', 'observed_label']].groupby('junction_seq')
    data_junction = pd.DataFrame(data_junction_group_by['observed_lfc'].nsmallest(n=4).groupby('junction_seq').mean())
    data_junction = data_junction.join(data_junction_group_by[['observed_label']].mean() >= 0.25)
    data_junction.rename(columns=dict(observed_lfc='junction_lfc', observed_label='junction_label'), inplace=True)
    data = data.set_index('junction_seq').join(data_junction).reset_index()

    # reduce to a junction-only dataset if requested
    if reduce:
        data.drop_duplicates('junction_seq', inplace=True)
        data['target_seq'] = data['junction_seq']
        data['observed_label'] = data['junction_label']
        data['observed_lfc'] = data['junction_lfc']

    return data


def append_rbp_predictions(data, suffix, context_5p=0, context_3p=0, reduce=False):
    # add RNA-prot scores
    data_rbp = pd.read_pickle(os.path.join('data-processed', 'junction-rbp-' + suffix + '.bz2'))
    rbp_list = data_rbp.columns.to_list()
    data['junction_id'] = data['guide_id'].apply(lambda s: s.split('.')[0])
    data = pd.merge(data, data_rbp, how='left', on='junction_id')

    # trim any per-nucleotide RBP predictions to target + context
    if data[rbp_list].dtypes.unique().tolist() != [float]:
        dictionary = data.to_dict('records')
        for i, row in enumerate(dictionary):
            start = row['junction_seq'].index(row['target_seq']) - context_5p
            length = context_5p + len(row['target_seq']) + context_3p
            for key in rbp_list:
                row[key] = row[key][start:start + length]
        data = pd.DataFrame(dictionary)

    # reduce RBP dimension if requested
    if reduce:
        num_peaks = np.array(data[rbp_list[0]].to_numpy().tolist())
        for rbp in rbp_list[1:]:
            num_peaks += np.array(data[rbp].to_numpy().tolist())
        data['num_peaks'] = num_peaks.tolist()
        rbp_list = ['num_peaks']

    return data, rbp_list
