import numpy as np
import pandas as pd
from data import LFC_COLS


def get_normalization_object(method: str):
    if method == 'No':
        return NoNormalization
    elif method == 'FrequentistQuantile':
        return FrequentistQuantileNormalization
    elif method == 'UnitVariance':
        return UnitVarianceNormalization
    elif method == 'ZeroMeanUnitVariance':
        return ZeroMeanUnitVarianceNormalization
    else:
        raise NotImplementedError


class LocationScaleNormalization(object):

    def __init__(self):
        self.params = None

    def normalize(self, df_data: pd.DataFrame):
        df_data = df_data.join(self.params, on='gene')
        for col in ['target_lfc', 'lfc_r1', 'lfc_r2', 'lfc_r3']:
            if col in df_data.columns:
                df_data[col] = (df_data[col] - df_data['location']) / df_data['scale']
        del df_data['location']
        del df_data['scale']
        return df_data

    def denormalize(self, df_tap: pd.DataFrame):
        df_tap = df_tap.set_index('gene').join(self.params, on='gene').reset_index()
        for col in ['target_lfc', 'target_pm_lfc', 'predicted_lfc', 'predicted_pm_lfc']:
            if col in df_tap.columns:
                df_tap[col] = df_tap[col] * df_tap['scale'] + df_tap['location']
        del df_tap['location']
        del df_tap['scale']
        return df_tap


class NoNormalization(LocationScaleNormalization):
    def __init__(self, data: pd.DataFrame):
        super().__init__()

        # compute location and scale from PM guides
        data = data[data.guide_type == 'PM'].groupby('gene')
        self.params = pd.DataFrame(data[LFC_COLS].apply(lambda _: 0.0).rename('location'))
        self.params = self.params.join(data[LFC_COLS].apply(lambda _: 1.0).rename('scale'))


class FrequentistQuantileNormalization(LocationScaleNormalization):

    def __init__(self, data: pd.DataFrame, q_loc: int = 50, q_neg: int = 10, q_pos: int = 90):
        assert q_neg < q_loc < q_pos
        super().__init__()

        # derive requisite quantiles for each gene
        data = data[data.guide_type == 'PM'].groupby('gene')[LFC_COLS]
        df = pd.DataFrame(data.apply(lambda x: np.nanquantile(x, q_loc / 100)).rename('location'))
        df = df.join(data.apply(lambda x: np.nanquantile(x, q_neg / 100)).rename('negative scale'))
        df = df.join(data.apply(lambda x: np.nanquantile(x, q_pos / 100)).rename('positive scale'))

        # finalize and save parameters
        df['scale'] = df['positive scale'] - df['negative scale']
        self.params = df[['location', 'scale']]


class UnitVarianceNormalization(LocationScaleNormalization):

    def __init__(self, data: pd.DataFrame):
        super().__init__()

        # location and positive scale are derived from non-targeting data
        self.params = pd.DataFrame(data={'location': 0}, index=data['gene'].unique())

        # negative scale derived for each gene
        data = data[data.guide_type == 'PM'].groupby('gene')[LFC_COLS]
        data = data.apply(lambda x: np.nanstd(x))
        self.params = self.params.join(data.rename('scale'))


class ZeroMeanUnitVarianceNormalization(LocationScaleNormalization):
    def __init__(self, data: pd.DataFrame):
        super().__init__()

        # compute location and scale from PM guides
        data = data[data.guide_type == 'PM'].groupby('gene')
        self.params = pd.DataFrame(data[LFC_COLS].apply(np.nanmean).rename('location'))
        self.params = self.params.join(data[LFC_COLS].apply(np.nanstd).rename('scale'))
