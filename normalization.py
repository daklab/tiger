import numpy as np
import pandas as pd

NORMALIZATION_COLUMNS = ['target_lfc', 'target_pm_lfc', 'predicted_lfc', 'predicted_pm_lfc']


def get_normalization_object(method: str):
    if method == 'No':
        return NoNormalization
    elif method == 'FrequentistQuantile':
        return FrequentistQuantileNormalization
    elif method == 'UnitInterval':
        return UnitIntervalNormalization
    elif method == 'UnitVariance':
        return UnitVarianceNormalization
    elif method == 'ZeroMeanUnitVariance':
        return ZeroMeanUnitVarianceNormalization
    elif method == 'DepletionRatio':
        return DepletionRatioNormalization
    elif method == 'Sigmoid':
        return SigmoidNormalization
    elif method == 'Tiger':
        return TigerNormalization
    else:
        raise NotImplementedError


class NoNormalization(object):
    def __init__(self, data: pd.DataFrame):
        self.output_fn = 'linear'

    @staticmethod
    def normalize(data: pd.DataFrame, **kwargs):
        return data

    @staticmethod
    def denormalize(targets_and_predictions: pd.DataFrame, **kwargs):
        return targets_and_predictions


class LocationScaleNormalization(object):

    def __init__(self):
        self.output_fn = 'linear'
        self.params = None

    def normalize(self, df_data: pd.DataFrame, cols: list = None):
        for gene in df_data['gene'].unique():
            for col in cols or NORMALIZATION_COLUMNS:
                if col in df_data.columns:
                    lfc = df_data.loc[df_data.gene == gene, col].to_numpy()
                    lfc = (lfc - self.params.loc[gene, 'location']) / self.params.loc[gene, 'scale']
                    df_data.loc[df_data.gene == gene, col] = lfc

        return df_data

    def denormalize(self, df_tap: pd.DataFrame, cols: list = None):
        for gene in df_tap['gene'].unique():
            for col in cols or NORMALIZATION_COLUMNS:
                if col in df_tap.columns:
                    lfc = df_tap.loc[df_tap.gene == gene, col].to_numpy()
                    lfc = lfc * self.params.loc[gene, 'scale'] + self.params.loc[gene, 'location']
                    df_tap.loc[df_tap.gene == gene, col] = lfc

        return df_tap


class FrequentistQuantileNormalization(LocationScaleNormalization):

    def __init__(self, data: pd.DataFrame, *, q_loc: int, q_neg: int, q_pos: int):
        assert q_neg < q_loc < q_pos
        super().__init__()

        # derive requisite quantiles for each gene
        data = data[data.guide_type == 'PM'].groupby('gene')['target_lfc']
        df = pd.DataFrame(data.apply(lambda x: np.nanquantile(x, q_loc / 100)).rename('location'))
        df = df.join(data.apply(lambda x: np.nanquantile(x, q_neg / 100)).rename('negative scale'))
        df = df.join(data.apply(lambda x: np.nanquantile(x, q_pos / 100)).rename('positive scale'))

        # finalize and save parameters
        df['scale'] = df['positive scale'] - df['negative scale']
        self.params = df[['location', 'scale']]


class UnitIntervalNormalization(LocationScaleNormalization):

    def __init__(self, data: pd.DataFrame, *, q_neg: int, q_pos: int, squash: bool):
        assert q_neg < q_pos
        assert isinstance(squash, bool)
        super().__init__()
        self.output_fn = 'sigmoid'
        self.squash = squash

        # derive requisite quantiles for each gene
        data = data[data.guide_type == 'PM'].groupby('gene')['target_lfc']
        df = pd.DataFrame(data.apply(lambda x: np.nanquantile(x, q_neg / 100)).rename('location'))
        df = df.join(data.apply(lambda x: np.nanquantile(x, q_neg / 100)).rename('negative scale'))
        df = df.join(data.apply(lambda x: np.nanquantile(x, q_pos / 100)).rename('positive scale'))

        # finalize and save parameters
        df['scale'] = df['positive scale'] - df['negative scale']
        self.params = df[['location', 'scale']]

    def normalize(self, df_data: pd.DataFrame, cols: list = None):
        df_data = super().normalize(df_data)
        if self.squash:
            for col in cols or NORMALIZATION_COLUMNS:
                if col in df_data.columns:
                    df_data[col] = np.clip(df_data[col], a_min=0, a_max=1)
        return df_data


class UnitVarianceNormalization(LocationScaleNormalization):

    def __init__(self, data: pd.DataFrame):
        super().__init__()

        # location and positive scale are derived from non-targeting data
        self.params = pd.DataFrame(data={'location': 0}, index=data['gene'].unique())

        # negative scale derived for each gene
        data = data[data.guide_type == 'PM'].groupby('gene')['target_lfc']
        data = data.apply(lambda x: np.nanstd(x))
        self.params = self.params.join(data.rename('scale'))


class ZeroMeanUnitVarianceNormalization(LocationScaleNormalization):
    def __init__(self, data: pd.DataFrame):
        super().__init__()

        # compute location and scale from PM guides
        data = data[data.guide_type == 'PM'].groupby('gene')
        self.params = pd.DataFrame(data['target_lfc'].apply(np.nanmean).rename('location'))
        self.params = self.params.join(data['target_lfc'].apply(np.nanstd).rename('scale'))


class DepletionRatioNormalization(object):
    def __init__(self, data: pd.DataFrame):
        self.output_fn = 'softplus'
        pass

    @staticmethod
    def normalize(data: pd.DataFrame, cols: list = None):
        for col in cols or NORMALIZATION_COLUMNS:
            if col in data.columns:
                data[col] = 2 ** data[col]
        return data

    @staticmethod
    def denormalize(targets_and_predictions: pd.DataFrame, cols: list = None):
        for col in cols or NORMALIZATION_COLUMNS:
            if col in targets_and_predictions.columns:
                targets_and_predictions[col] = np.log2(targets_and_predictions[col])
        return targets_and_predictions


class SigmoidNormalization(object):

    def __init__(self, data: pd.DataFrame, *, min_point: float, cutoff_point: float):
        super().__init__()
        assert 0 < min_point < cutoff_point < 1
        self.output_fn = 'sigmoid'

        # location and positive scale are derived from non-targeting data
        self.cutoff = data.loc[data.target_label == 1, 'target_lfc'].max() / 2 + \
                      data.loc[data.target_label == 0, 'target_lfc'].min() / 2

        # parameterize a sigmoid function for each gene
        self.params = pd.DataFrame()
        for gene in data['gene'].unique():
            x = np.array([[data.loc[data.gene == gene, 'target_lfc'].min()], [self.cutoff]])
            x = np.concatenate([x, np.ones_like(x)], axis=1)
            y = np.log(np.array([[min_point], [cutoff_point]]) ** -1 - 1)
            params = dict(zip(('a', 'b'), np.squeeze(np.linalg.inv(x.T @ x) @ x.T @ y)))
            self.params = pd.concat([self.params, pd.DataFrame(params, index=[gene])])

    def normalize(self, data: pd.DataFrame, cols: list = None):
        for gene in data['gene'].unique():
            a, b = self.params.loc[gene, ['a', 'b']]
            for col in cols or NORMALIZATION_COLUMNS:
                if col in data.columns:
                    x = data.loc[data.gene == gene, col].to_numpy()
                    data.loc[data.gene == gene, col] = 1 / (1 + np.exp(a * x + b))
        return data

    def denormalize(self, data: pd.DataFrame, cols: list = None):
        for gene in data['gene'].unique():
            a, b = self.params.loc[gene, ['a', 'b']]
            for col in cols or NORMALIZATION_COLUMNS:
                if col in data.columns:
                    x = data.loc[data.gene == gene, col].to_numpy()
                    data.loc[data.gene == gene, col] = (np.log(1 / x - 1) - b) / a
        return data


class TigerNormalization(object):

    def __init__(self, data: pd.DataFrame, *, active_sat_val: int = 0.05, inactive_sat_val: int = 0.95):
        assert 0.05 <= active_sat_val < inactive_sat_val <= 0.95
        self.output_fn = 'sigmoid'

        # inactive saturation point in LFC space
        self.inactive_sat_val = inactive_sat_val
        self.inactive_sat_lfc = data.loc[data.target_label == 1, 'target_lfc'].max() / 2 + \
                                data.loc[data.target_label == 0, 'target_lfc'].min() / 2

        # active saturation point in LFC space (per-gene)
        self.active_sat_val = active_sat_val
        self.active_sat_lfc = data.loc[data.guide_type == 'PM'].groupby('gene')['target_lfc'].quantile(active_sat_val)

        # parameters
        self.slope = (self.inactive_sat_val - self.active_sat_val) / (self.inactive_sat_lfc - self.active_sat_lfc)
        self.intercept = -self.active_sat_lfc * self.slope + self.active_sat_val
        self.a_active = self.slope / self.active_sat_val
        self.b_active = self.a_active * self.active_sat_lfc - np.log(self.active_sat_val)
        self.a_inactive = self.slope / (1 - self.inactive_sat_val)
        self.b_inactive = -self.a_inactive * self.inactive_sat_lfc - np.log(1 - self.inactive_sat_val)

    def normalize(self, data: pd.DataFrame, cols: list = None):
        for gene in data['gene'].unique():
            for col in cols or NORMALIZATION_COLUMNS:
                if col in data.columns:
                    lfc = data.loc[data.gene == gene, col].to_numpy()

                    # regime indices
                    active_sat = lfc < self.active_sat_lfc[gene]
                    linear_regime = (self.active_sat_lfc[gene] <= lfc) & (lfc <= self.inactive_sat_lfc)
                    inactive_sat = self.inactive_sat_lfc < lfc

                    # regime maps
                    lfc[active_sat] = np.exp(self.a_active[gene] * lfc[active_sat] - self.b_active[gene])
                    lfc[linear_regime] = self.slope[gene] * lfc[linear_regime] + self.intercept[gene]
                    lfc[inactive_sat] = 1 - np.exp(-self.a_inactive[gene] * lfc[inactive_sat] - self.b_inactive[gene])

                    data.loc[data.gene == gene, col] = lfc

        return data

    def denormalize(self, df_tap: pd.DataFrame, cols: list = None):
        for gene in df_tap['gene'].unique():
            for col in cols or NORMALIZATION_COLUMNS:
                if col in df_tap.columns:
                    score = df_tap.loc[df_tap.gene == gene, col].to_numpy()

                    # regime indices
                    active_sat = score < self.active_sat_val
                    linear_regime = (self.active_sat_val <= score) & (score <= self.inactive_sat_val)
                    inactive_sat = self.inactive_sat_val < score

                    # regime maps
                    score[active_sat] = (np.log(score[active_sat]) + self.b_active[gene]) / self.a_active[gene]
                    score[linear_regime] = (score[linear_regime] - self.intercept[gene]) / self.slope[gene]
                    score[inactive_sat] = -(np.log1p(-score[inactive_sat]) + self.b_inactive[gene]) / self.a_active[gene]

                    df_tap.loc[df_tap.gene == gene, col] = score

        return df_tap

