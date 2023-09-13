import argparse
import glob
import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from data import sequence_complement, INDEX_COLS, LFC_COLS, SEQUENCE_FEATS, SCALAR_FEATS

# relevant columns
GUIDE_COLS = ['end', 'start', 'junction_dist_5p', 'junction_dist_3p', 'guide_type', 'guide_seq']
TRANSCRIPT_COLS = ['target_seq', 'length', 'cds_start', 'cds_stop']
COLUMNS_TO_SAVE = INDEX_COLS + ['guide_type'] + LFC_COLS + SEQUENCE_FEATS + SCALAR_FEATS


def process_raw_wessels_data(sub_dir, num_folds, seed):
    """
    Process raw data from either Wessels et al. 2020 or 2022 and save the processed data as a Pandas DataFrame
    :param sub_dir: a subdirectory containing raw data from one of the Wessels et al. papers
    :param num_folds: number of validation folds
    :param seed: random number seed
    :return: None
    """
    # prepare guide file
    df_guide = pd.read_csv(os.path.join('data-raw', sub_dir, 'guide_table.txt'), delimiter='\t')
    df_guide.rename(columns={
        'TargetGene': 'gene',
        'GuideID': 'guide_id',
        'GuideSeq': 'guide_seq',
        'Type': 'guide_type',
        'Dist2Junction_5p': 'junction_dist_5p',
        'Dist2Junction_3p': 'junction_dist_3p'
    }, inplace=True)
    df_guide = df_guide[df_guide.gene != 'NT']
    df_guide = df_guide[df_guide.guide_type != 'NT']
    df_guide['guide_seq'] = df_guide['guide_seq'].apply(lambda seq: seq[::-1])
    for int_col in ['end', 'start', 'junction_dist_5p', 'junction_dist_3p']:
        df_guide[int_col] = df_guide[int_col].astype(int)
    df_guide = df_guide[INDEX_COLS + GUIDE_COLS]
    df_guide.set_index(INDEX_COLS, inplace=True)

    # prepare LFC file
    df_lfc = pd.DataFrame()
    for lfc_csv in glob.glob(os.path.join('data-raw', sub_dir, 'lfc', '*.csv')):
        df_lfc = pd.concat([df_lfc, pd.read_csv(lfc_csv)])
    df_lfc.rename(columns={'D30_R1': 'lfc_r1', 'D30_R2': 'lfc_r2', 'D30_R3': 'lfc_r3'}, inplace=True)
    df_lfc['gene'] = [s.split('_')[0] for s in df_lfc.index.values]
    df_lfc['guide_id'] = ['_'.join(s.split('_')[1:]) for s in df_lfc.index.values]
    df_lfc.reset_index(drop=True, inplace=True)
    df_lfc = df_lfc[INDEX_COLS + LFC_COLS]
    df_lfc.set_index(INDEX_COLS, inplace=True)

    # save non-targeting data
    try:
        df_nt = df_lfc.loc['NT', :].reset_index()[LFC_COLS]
        df_nt.to_pickle(os.path.join('data-processed', sub_dir + '-nt.bz2'))
    except KeyError:
        print(sub_dir, 'has no non-targeting guides!')

    # prepare transcript file
    df_transcripts = pd.read_csv(os.path.join('data-raw', sub_dir, 'transcript_lut.tsv'), delimiter='\t')
    df_transcripts.rename(columns={'gene_name': 'gene'}, inplace=True)
    df_transcripts['cds_start'] = df_transcripts['cds_coord'].apply(lambda s: s.split('-')[0]).astype(int)
    df_transcripts['cds_stop'] = df_transcripts['cds_coord'].apply(lambda s: s.split('-')[1]).astype(int)
    df_transcripts = load_transcripts(df_transcripts, os.path.join('data-raw', sub_dir, 'transcripts'), '.fasta')
    df_transcripts = df_transcripts[['gene'] + TRANSCRIPT_COLS]
    df_transcripts.set_index('gene', inplace=True)

    # load and join folding energy features
    df_fold_mfe = pd.read_csv(os.path.join('data-raw', sub_dir, 'features', 'foldMFE.txt'), delimiter='\t')
    del df_fold_mfe['Guide_Name']
    df_fold_mfe.rename(columns={
        'TargetGene': 'gene',
        'GuideID': 'guide_id',
        'MFE': 'mfe',
        'DR': 'direct_repeat',
        'Gquad': 'g_quad',
        'Fold': 'guide_ss',
    }, inplace=True)
    df_fold_mfe.set_index(INDEX_COLS, inplace=True)

    # load and join hybridization energy features
    df_hyb_mfe = pd.read_csv(os.path.join('data-raw', sub_dir, 'features', 'hybMFE.txt'), delimiter='\t')
    del df_hyb_mfe['Guide_Name']
    df_hyb_mfe.rename(columns={
        'TargetGene': 'gene',
        'GuideID': 'guide_id',
        'hybMFE_1.23': 'hybrid_mfe_1_23',
        'hybMFE_15.9': 'hybrid_mfe_15_9',
        'hybMFE_3.12': 'hybrid_mfe_3_12',
    }, inplace=True)
    df_hyb_mfe.set_index(INDEX_COLS, inplace=True)

    # target site accessibility features
    df_target_access = pd.read_csv(os.path.join('data-raw', sub_dir, 'features', 'targetAccess.txt'), delimiter='\t')
    del df_target_access['Guide_Name']
    df_target_access.rename(columns={
        'TargetGene': 'gene',
        'GuideID': 'guide_id',
        'Log10_Unpaired': 'log_unpaired',
        'Log10_Unpaired_p11': 'log_unpaired_11',
        'Log10_Unpaired_p19': 'log_unpaired_19',
        'Log10_Unpaired_p25': 'log_unpaired_25',
    }, inplace=True)
    df_target_access.set_index(['gene', 'guide_id'], inplace=True)

    # merge tables
    assert not df_guide.index.has_duplicates
    assert not df_lfc.index.has_duplicates
    assert not df_transcripts.index.has_duplicates
    df_scalar_features = df_fold_mfe.join(df_hyb_mfe, how='inner')
    df_scalar_features = df_scalar_features.join(df_target_access, how='inner')

    # aggregate the data
    df_data = df_guide.join(df_lfc, how='inner')
    df_data = df_data.join(df_transcripts, how='left')
    df_data = df_data.join(df_scalar_features, how='inner')

    # cut away non-targeted gene sequence, while keeping nearby sequence context
    for index, row in df_data.iterrows():
        df_data.loc[index, 'target_seq'] = row['target_seq'][row['end'] - 1:row['start']]
        df_data.loc[index, '5p_context'] = row['target_seq'][:row['end'] - 1]
        df_data.loc[index, '3p_context'] = row['target_seq'][row['start']:]

    # add normalized location of target site along transcript
    guide_loc = df_data[['start', 'end']].mean(axis=1)
    df_data['loc_utr_5p'] = guide_loc / (df_data['cds_start'] - 1)
    df_data['loc_cds'] = (guide_loc - df_data['cds_start']) / (df_data['cds_stop'] - df_data['cds_start'])
    df_data['loc_utr_3p'] = (guide_loc - df_data['cds_stop'] + 1) / (df_data['length'] - df_data['cds_stop'] + 1)
    for location_float in ['loc_utr_5p', 'loc_cds', 'loc_utr_3p']:
        df_data[location_float] = np.clip(df_data[location_float], a_min=0.0, a_max=1.0)
    df_data['log_gene_len'] = np.log10(df_data['length'])

    # keep only the columns of interest
    df_data.reset_index(inplace=True)
    keep_columns = [col for col in COLUMNS_TO_SAVE if col in df_data.columns]
    missing_columns = list(set(COLUMNS_TO_SAVE) - set(keep_columns))
    missing_columns.sort()
    print(sub_dir, 'data is missing:', missing_columns)
    df_data = df_data[keep_columns]

    # add fold assignments and save data
    df_data = fold_assignments(df_data, num_folds, seed)
    df_data.to_pickle(os.path.join('data-processed', sub_dir + '.bz2'))


def process_hap_titration_data(num_folds, seed):

    # load the data
    data_dir = os.path.join('data-raw', 'hap-titration')
    data = pd.read_csv(os.path.join(data_dir, 'Titration.GuideInfo.txt'), sep='\t').set_index('UID')
    lfc = pd.read_csv(os.path.join(data_dir, 'Titration.L2FCs.txt'), sep='\t')
    data = data.join(lfc[['HAP1_D14_R1', 'HAP1_D14_R2', 'HAP1_D14_R3']]).reset_index()

    # rename columns
    data = data.rename(columns={
        'GeneName': 'gene',
        'UID': 'guide_id',
        'Type': 'guide_type',
        'HAP1_D14_R1': 'lfc_r1',
        'HAP1_D14_R2': 'lfc_r2',
        'HAP1_D14_R3': 'lfc_r3',
        'Dist2Junction_5p': 'junction_dist_5p',
        'Dist2Junction_3p': 'junction_dist_3p',
        'hybrid_mfe_1_23': 'hybrid_mfe_1_23',
        'hybMFE_15.9': 'hybrid_mfe_15_9',
        'hybMFE_3.12': 'hybrid_mfe_3_12',
        'log_unpaired': 'log_unpaired',
        'log10_unpaired_p11': 'log_unpaired_11',
        'log10_unpaired_p19': 'log_unpaired_19',
        'log10_unpaired_p25': 'log_unpaired_25',
    })

    # save non-targeting data and remove it
    data_nt = data[data.guide_type == 'NT'].reset_index()[LFC_COLS]
    data_nt.to_pickle(os.path.join('data-processed', 'hap-titration-nt.bz2'))
    data = data[data.guide_type != 'NT']

    # sequence features
    data['guide_seq'] = data['GuideSeq'].apply(lambda seq: seq[::-1])
    data['5p_context'] = data['TargetSeqContext'].apply(lambda seq: seq[:2])
    data['target_seq'] = data['TargetSeqContext'].apply(lambda seq: seq[2:-2])
    data['3p_context'] = data['TargetSeqContext'].apply(lambda seq: seq[-2:])

    # target location
    data['loc_utr_5p'] = data['loc'].apply(lambda loc: loc[1:-1].split(';')[0])
    data['loc_cds'] = data['loc'].apply(lambda loc: loc[1:-1].split(';')[1])
    data['loc_utr_3p'] = data['loc'].apply(lambda loc: loc[1:-1].split(';')[2])

    # keep only the columns of interest
    data.reset_index(inplace=True)
    keep_columns = [col for col in COLUMNS_TO_SAVE if col in data.columns]
    missing_columns = list(set(COLUMNS_TO_SAVE) - set(keep_columns))
    missing_columns.sort()
    print('HAP titration data is missing:', missing_columns)
    data = data[keep_columns]

    # add fold assignments and save data
    data = fold_assignments(data, num_folds, seed)
    data.to_pickle(os.path.join('data-processed', 'hap-titration.bz2'))


def process_raw_junction_data(num_folds, seed):
    """
    Process raw junction data and save the processed data as a Pandas DataFrame
    :param num_folds: number of validation folds
    :param seed: random number seed
    :return: None
    """
    # base directory
    base_dir = os.path.join('data-raw', 'junction')

    # process non-targeting data
    data_nt = os.path.join(base_dir, '220301_R1_remove_TechPool_wno_batch_LFC_plus_a375_gene_expression.txt.gz')
    data_nt = pd.read_csv(data_nt, sep='\t', low_memory=False).rename(columns={'sgrna': 'guide_id'})
    data_nt = data_nt[data_nt.day == 'D21']
    data_nt = data_nt.pivot(index=['guide_id', 'type'], columns='replicate', values='logFC').reset_index('type')
    data_nt = data_nt.rename(columns={'R2': 'lfc_r2', 'R3': 'lfc_r3'})
    data_nt['lfc_r1'] = np.nan
    data_nt = data_nt.loc[data_nt.type == 'NT', LFC_COLS]
    data_nt.to_pickle(os.path.join('data-processed', 'junction-nt.bz2'))

    # load data
    data = os.path.join(base_dir, '230411_final_screen_table_essential_genes.txt')
    data = pd.read_csv(data, delimiter='\t', low_memory=False)
    data = data[data.day == 'D21']
    assert set(data['type'].unique()) == {'essential'}
    num_guides = data['guide_id'].nunique()

    # adjust and rename existing features
    data['guide_seq'] = data['guide_sequence'].apply(lambda seq: seq[::-1])
    data['guide_type'] = 'PM'
    data['target_seq'] = data['target_sequence'].apply(lambda seq: seq[5:-5])
    data['5p_context'] = data['target_sequence'].apply(lambda seq: seq[:5])
    data['3p_context'] = data['target_sequence'].apply(lambda seq: seq[-5:])
    data['log_gene_len'] = np.log10(data['length'])
    data['strand'] = data['strand'].apply(lambda s: +1 if s == '+' else -1)
    data['junction_id'] = data['screen_junc_id']
    data['junction_olap_5p'] = (data['guide_start'] - data['junc_start']) / data['guide_sequence'].apply(len)
    data['junction_olap_3p'] = (data['guide_end'] - data['junc_end']) / data['guide_sequence'].apply(len)
    data['perc_gene_nuc'] = data['perc.gene.nuc'].apply(lambda x: x / 100)
    data['perc_junc_nuc'] = data['perc.junc.nuc'].apply(lambda x: x / 100)
    data.rename(columns={
        'gene_name': 'gene',
        'DR': 'direct_repeat',
        'Gquad': 'g_quad',
        'MFE': 'mfe',
        'hybMFE_3.12': 'hybrid_mfe_3_12',
        'hybMFE_15.9': 'hybrid_mfe_15_9',
        'Log10_Unpaired': 'log_unpaired',
    }, inplace=True)

    # pivot and rejoin replicate LFC values
    lfc = data[['guide_id', 'replicate', 'logFC']].set_index('guide_id')
    lfc = lfc.pivot(columns='replicate', values='logFC').rename(columns={'R2': 'lfc_r2', 'R3': 'lfc_r3'})
    data['lfc_r1'] = np.nan
    assert not lfc.index.has_duplicates
    data = data.set_index('guide_id')
    data = data.loc[data.index.duplicated(keep='first')]
    data = data.join(lfc)
    assert not data.index.has_duplicates
    data = data.reset_index()

    # add junction sequence
    junction_sequence = pd.read_csv(os.path.join(base_dir, 'junc_seq.txt'), delimiter='\t', low_memory=False)
    junction_sequence.rename(columns={'junc.name': 'junction_id', 'junc.sequence': 'junction_seq'}, inplace=True)
    data = data.set_index('junction_id').join(junction_sequence.set_index('junction_id')['junction_seq']).reset_index()
    for index, row in data.iterrows():
        target_seq = row['5p_context'] + row['target_seq'] + row['3p_context']
        assert target_seq in row['junction_seq']

    # rename remaining columns
    keep_columns = [col for col in COLUMNS_TO_SAVE if col in data.columns]
    keep_columns += ['junction_id', 'junction_seq', 'junction_category']
    missing_columns = list(set(COLUMNS_TO_SAVE) - set(keep_columns))
    missing_columns.sort()
    print('Junction data is missing:', missing_columns)
    num_perc_junc_nuc_nan = sum(data['perc_junc_nuc'].isna())
    data = data.loc[~data[list(set(keep_columns) - {'lfc_r1'})].isna().any(axis=1), keep_columns]
    assert len(data) == num_guides - num_perc_junc_nuc_nan

    # add fold assignments and save data
    data = fold_assignments(data, num_folds, seed)
    data.to_pickle(os.path.join('data-processed', 'junction.bz2'))


def process_raw_junction_rbp_data():
    rna_prot_dir = os.path.join('data-raw/junction/RNAprot')

    # RNA prot junction-level predictions
    df_rbp_junc = pd.read_csv(os.path.join(rna_prot_dir, 'output_averaged.csv'))
    df_rbp_junc = df_rbp_junc.rename(columns=dict(site_id='junction_id')).set_index('junction_id')
    df_rbp_junc.to_pickle(os.path.join('data-processed', 'junction-rbp-junc.bz2'))

    # RNA prot junction-level predictions
    for (file, suffix) in zip(['all_peak_outputs.csv', 'relaxed_peak_outputs.csv'], ['nt', 'nt_relaxed']):
        df_rbp_nt = pd.read_csv(os.path.join(rna_prot_dir, file))
        df_rbp_nt = df_rbp_nt.rename(columns=dict(ref_id='junction_id'))
        if file == 'all_peak_outputs.csv':
            df_rbp_nt['RBP'] = df_rbp_nt['Cell_line'] + '_' + df_rbp_nt['Gene']
        elif file == 'relaxed_peak_outputs.csv':
            df_rbp_nt.rename(columns=dict(rbp='RBP'), inplace=True)
        else:
            raise NotImplementedError
        df_peak_s = df_rbp_nt.pivot_table(index='junction_id', columns='RBP', values='peak_region_s', fill_value=101)
        df_peak_e = df_rbp_nt.pivot_table(index='junction_id', columns='RBP', values='peak_region_e', fill_value=100)
        peaks = np.zeros(df_peak_s.shape + (101,), dtype=np.int8)
        np.put_along_axis(arr=peaks, indices=df_peak_s.values[..., None].astype(int) - 1, values=1, axis=-1)
        np.put_along_axis(arr=peaks, indices=df_peak_e.values[..., None].astype(int), values=-1, axis=-1)
        peaks = np.cumsum(peaks, axis=-1)[..., :100]
        df_rbp_nt = pd.DataFrame(index=df_peak_s.index, columns=df_peak_s.columns, data=peaks.tolist())
        df_rbp_nt.to_pickle(os.path.join('data-processed', 'junction-rbp-' + suffix + '.bz2'))


def process_all_junctions():

    # sequence for all junctions
    data = pd.read_csv('data-raw/junction/gencode_v41_all_junction_guides.txt', delimiter='\t')
    data = data.rename(columns=dict( gene_name='gene', guide_sequence='guide_seq', target_sequence='target_seq'))
    data['guide_type'] = 'PM'
    data['5p_context'] = data['target_seq'].apply(lambda s: s[:5])
    data['3p_context'] = data['target_seq'].apply(lambda s: s[-5:])
    data['target_seq'] = data['target_seq'].apply(lambda s: s[5:-5])
    assert set(data['guide_seq'].apply(len).unique()) == set(data['target_seq'].apply(len).unique())
    data = data.set_index(INDEX_COLS)
    data = data[[col for col in COLUMNS_TO_SAVE if col in data.columns]]
    data.to_pickle(os.path.join('data-processed', 'junction-all.bz2'))


def process_raw_hap_validation_data():
    """
    Process raw junction isoform data and save the processed data as a Pandas DataFrame
    :return: None
    """
    # hap validation guides
    data_file = os.path.join('data-raw', 'hap-validation', 'validation_guides.csv')
    df_data = pd.read_csv(data_file)
    df_data.rename(columns={'top_sequence': 'guide_seq'}, inplace=True)

    # prepare guide/target sequence
    df_data['guide_seq'] = df_data['guide_seq'].apply(lambda seq: seq[::-1])
    df_data['target_seq'] = df_data['guide_seq'].apply(sequence_complement)

    # stuff missing values
    df_data['gene'] = 'unknown'
    df_data['guide_type'] = 'PM'
    df_data[['5p_context', '3p_context']] = ''

    # finalize data
    keep_columns = [col for col in COLUMNS_TO_SAVE if col in df_data.columns]
    missing_columns = list(set(COLUMNS_TO_SAVE) - set(keep_columns))
    missing_columns.sort()
    print('hap=validation', 'data is missing:', missing_columns)
    df_data = df_data[keep_columns]

    # save data
    df_data.to_pickle(os.path.join('data-processed', 'hap-validation.bz2'))


def load_transcripts(df_transcripts, transcript_dir, file_type):
    """
    Load transcript sequence for each gene in the provided transcript table
    :param df_transcripts: DataFrame with transcript IDs that point to transcript files
    :param transcript_dir: directory containing transcript files
    :param file_type: FASTA file type postfix (e.g. .fasta or .fa)
    :return: df_transcripts with gene sequences loaded into additional column
    """
    df_transcripts['target_seq'] = None
    for index, row in df_transcripts.iterrows():
        with open(os.path.join(transcript_dir, row['transcript_id'] + file_type), 'r') as file:
            seq_records = [s for s in SeqIO.parse(file, 'fasta')]
            assert len(seq_records) == 1
            target_seq = str(seq_records[0].seq)
            assert len(target_seq) == row['length']
            df_transcripts.loc[index, 'target_seq'] = target_seq
    return df_transcripts


def fold_assignments(df_data, num_folds, seed):
    # set random number seed
    np.random.seed(seed)

    # guide folds
    df_data['guide_fold'] = 1 + np.random.choice(num_folds, len(df_data))

    # target folds
    df_pm = df_data[df_data.guide_type == 'PM'][['target_seq']]
    df_pm['target_fold'] = 1 + np.random.choice(num_folds, len(df_pm))
    df_data = pd.merge(df_data, df_pm, how='inner', on='target_seq')

    return df_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all', help='which dataset to process')
    parser.add_argument('--num_folds', type=int, default=10, help='number of folds')
    parser.add_argument('--seed', type=int, default=112358, help='random number seed for fold assignments')
    args = parser.parse_args()

    # make sure output directory exists
    os.makedirs('data-processed', exist_ok=True)

    # prepare requested data sources
    if args.dataset == 'all' or args.dataset == 'flow-cytometry':
        process_raw_wessels_data('flow-cytometry', args.num_folds, args.seed)
    if args.dataset == 'all' or args.dataset == 'off-target':
        process_raw_wessels_data('off-target', args.num_folds, args.seed)
    if args.dataset == 'all' or args.dataset == 'hap-titration':
        process_hap_titration_data(args.num_folds, args.seed)
    if args.dataset == 'all' or args.dataset == 'junction':
        process_raw_junction_data(args.num_folds, args.seed)
        process_raw_junction_rbp_data()
        process_all_junctions()
    # if args.dataset == 'all' or args.dataset == 'hap-validation':
    #     process_raw_hap_validation_data()
