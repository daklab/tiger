import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def get_normalization_object(method: str):
    if method == 'No':
        return NoNormalization
    elif method == 'FrequentistQuantile':
        return FrequentistQuantileNormalization
    elif method == 'UnitInterval':
        return UnitIntervalNormalization
    elif method == 'UnitVariance':
        return UnitVarianceNormalization
    elif method == 'UnitMeanOfSquares':
        return UnitMeanOfSquaresNormalization
    elif method == 'ZeroMeanUnitVariance':
        return ZeroMeanUnitVarianceNormalization
    elif method == 'DepletionRatio':
        return DepletionRatioNormalization
    elif method == 'QuantileMatching':
        return QuantileMatchingNormalization
    else:
        raise NotImplementedError


class Normalization(object):
    def __init__(self, df: pd.DataFrame):
        self.original_lfc = df[['gene', 'guide_seq', 'observed_lfc']].copy().set_index(['gene', 'guide_seq'])
        assert not self.original_lfc.index.has_duplicates

    def normalize_targets(self, df: pd.DataFrame):
        raise NotImplementedError

    def denormalize_observations(self, df: pd.DataFrame):
        # restore observed values
        if 'observed_lfc' in df.columns:
            df = pd.merge(
                left=df,
                right=self.original_lfc,
                left_on=['gene', 'guide_seq'],
                right_index=True,
                suffixes=('_normalized', '')
            )
        if 'observed_pm_lfc' in df.columns:
            df_pm = df.loc[df.guide_type == 'PM', ['gene', 'target_seq', 'observed_lfc']]
            df_pm = df_pm.rename(columns={'observed_lfc': 'observed_pm_lfc'}).set_index(['gene', 'target_seq'])
            df = pd.merge(
                left=df,
                right=df_pm,
                how='left',
                left_on=['gene', 'target_seq'],
                right_index=True,
                suffixes=('_normalized', '')
            )
        return df

    def denormalize_predictions(self, df: pd.DataFrame):
        df['predicted_lfc_normalized'] = df['predicted_lfc']
        if 'predicted_pm_lfc' in df.columns:
            df['predicted_pm_lfc_normalized'] = df['predicted_pm_lfc']
        return df

    def denormalize_targets_and_predictions(self, df: pd.DataFrame):
        return self.denormalize_predictions(self.denormalize_observations(df.copy()))


class NoNormalization(Normalization):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.output_fn = 'linear'

    def normalize_targets(self, df: pd.DataFrame):
        return df

    def denormalize_observations(self, df: pd.DataFrame):
        return df

    def denormalize_predictions(self, df: pd.DataFrame):
        return df


class LocationScaleNormalization(Normalization):

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.output_fn = 'linear'
        self.params = None

    def normalize_targets(self, df: pd.DataFrame):
        for gene in df['gene'].unique():
            for col in ['observed_lfc', 'observed_pm_lfc']:
                if col in df.columns:
                    lfc = df.loc[df.gene == gene, col].to_numpy()
                    lfc = (lfc - self.params.loc[gene, 'location']) / self.params.loc[gene, 'scale']
                    df.loc[df.gene == gene, col] = lfc
        return df

    def denormalize_predictions(self, df: pd.DataFrame):
        df = super().denormalize_predictions(df)
        for gene in df['gene'].unique():
            for col in ['predicted_lfc', 'predicted_pm_lfc']:
                if col in df.columns:
                    lfc = df.loc[df.gene == gene, col].to_numpy()
                    lfc = lfc * self.params.loc[gene, 'scale'] + self.params.loc[gene, 'location']
                    df.loc[df.gene == gene, col] = lfc
        return df


class FrequentistQuantileNormalization(LocationScaleNormalization):

    def __init__(self, df: pd.DataFrame, *, q_loc: int, q_neg: int, q_pos: int):
        assert q_neg < q_loc < q_pos
        super().__init__(df)

        # derive requisite quantiles for each gene
        df = df[df.guide_type == 'PM'].groupby('gene')['observed_lfc']
        loc = df.apply(lambda x: np.nanquantile(x, q_loc / 100)).rename('location')
        neg_scale = df.apply(lambda x: np.nanquantile(x, q_neg / 100)).rename('negative scale')
        pos_scale = df.apply(lambda x: np.nanquantile(x, q_pos / 100)).rename('positive scale')

        # finalize and save parameters
        params = pd.DataFrame([loc, neg_scale, pos_scale]).T
        params['scale'] = params['positive scale'] - params['negative scale']
        self.params = params[['location', 'scale']]


class UnitIntervalNormalization(LocationScaleNormalization):

    def __init__(self, df: pd.DataFrame, *, q_neg: int, q_pos: int, squash: bool):
        assert q_neg < q_pos
        assert isinstance(squash, bool)
        super().__init__(df)
        self.output_fn = 'sigmoid'
        self.squash = squash

        # derive requisite quantiles for each gene
        df = df[df.guide_type == 'PM'].groupby('gene')['observed_lfc']
        loc = df.apply(lambda x: np.nanquantile(x, q_neg / 100)).rename('location')
        pos = df.apply(lambda x: np.nanquantile(x, q_pos / 100)).rename('positive quant')

        # finalize and save parameters
        params = pd.DataFrame([loc, pos]).T
        params['scale'] = params['positive quant'] - params['location']
        self.params = params[['location', 'scale']]

    def normalize_targets(self, df: pd.DataFrame):
        df = super().normalize_targets(df)
        if self.squash:
            for col in ['observed_lfc', 'observed_pm_lfc']:
                if col in df.columns:
                    df[col] = np.clip(df[col], a_min=0, a_max=1)
        return df


class UnitVarianceNormalization(LocationScaleNormalization):

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.params = pd.DataFrame(data={'location': 0}, index=df['gene'].unique())
        df = df[df.guide_type == 'PM'].groupby('gene')['observed_lfc'].apply(np.nanstd)
        self.params = self.params.join(df.rename('scale'))


class UnitMeanOfSquaresNormalization(LocationScaleNormalization):

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.params = pd.DataFrame(data={'location': 0}, index=df['gene'].unique())
        df = df[df.guide_type == 'PM'].groupby('gene')['observed_lfc'].apply(lambda x: np.sqrt(np.nanmean(x*x)))
        self.params = self.params.join(df.rename('scale'))


class ZeroMeanUnitVarianceNormalization(LocationScaleNormalization):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        df = df[df.guide_type == 'PM'].groupby('gene')
        self.params = pd.DataFrame(df['observed_lfc'].apply(np.nanmean).rename('location'))
        self.params = self.params.join(df['observed_lfc'].apply(np.nanstd).rename('scale'))


class DepletionRatioNormalization(Normalization):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.output_fn = 'softplus'

    def normalize_targets(self, df: pd.DataFrame):
        for col in ['observed_lfc', 'observed_pm_lfc']:
            if col in df.columns:
                df[col] = 2 ** df[col]
        return df

    def denormalize_predictions(self, df: pd.DataFrame):
        df = super().denormalize_predictions(df)
        for col in ['predicted_lfc', 'predicted_pm_lfc']:
            if col in df.columns:
                df[col] = np.log2(df[col])
        return df


class QuantileMatchingNormalization(Normalization):
    def __init__(self, df: pd.DataFrame, debug: bool = False):
        super().__init__(df)
        self.output_fn = 'linear'

        # set quantile resolution
        quantiles = np.concatenate([np.arange(0.0, 1 + 1e-6, .005)])

        # identify reference gene (i.e. median essentiality) and grab its LFC's quantiles
        reference_gene = df.loc[df.guide_type == 'PM'].groupby('gene')['observed_label'].mean()
        reference_gene = reference_gene.sort_values().iloc[:-2].index.values[-1]
        reference_lfc = df.loc[(df.guide_type == 'PM') & (df.gene == reference_gene), 'observed_lfc']
        self.y = np.array([np.quantile(reference_lfc, q) for q in quantiles])

        # parametric quantile matching function for each gene
        self.params = pd.DataFrame()
        for gene in df['gene'].unique():
            lfc = df.loc[(df.guide_type == 'PM') & (df.gene == gene), 'observed_lfc']
            x = np.array([np.quantile(lfc, q) for q in quantiles])
            self.params = pd.concat([self.params, pd.DataFrame(x[None, ...], index=[gene])])

            # debugging plots
            if debug:
                fig, ax = plt.subplots(2)
                fig.suptitle(gene)
                lfc_sweep = np.linspace(min(lfc), max(lfc), 1000)
                gp_output = self.gaussian_process(lfc_sweep, self.params.loc[gene].values, self.y)
                sns.lineplot(x=lfc_sweep, y=gp_output, ax=ax[0])
                x_normalized = self.gaussian_process(x, self.params.loc[gene].values, self.y)
                x_denormalized = self.gaussian_process(x_normalized, self.y, self.params.loc[gene].values)
                sns.scatterplot(x=x, y=self.y, ax=ax[1])
                sns.scatterplot(x=x_normalized, y=self.y, ax=ax[1])
                sns.scatterplot(x=x_denormalized, y=self.y, ax=ax[1])
                plt.plot([min(self.y), max(self.y)], [min(self.y), max(self.y)], color='k')
                plt.show()

    @staticmethod
    def gaussian_process(z, x, y, length_scale: float = 2.0, noise_variance: float = 1e-1):
        kxx = np.exp(-(x[:, None] - x[None, :]) ** 2 / 2 / length_scale)
        kzx = np.exp(-(z[:, None] - x[None, :]) ** 2 / 2 / length_scale)
        f = kzx @ np.linalg.inv(kxx + noise_variance * np.eye(len(x))) @ y[:, None]
        return np.squeeze(f)

    def normalize_targets(self, df: pd.DataFrame):
        for gene in df['gene'].unique():
            params = self.params.loc[gene].values
            for col in ['observed_lfc', 'observed_pm_lfc']:
                if col in df.columns:
                    df.loc[df.gene == gene, col] = self.gaussian_process(df.loc[df.gene == gene, col], params, self.y)
                    df.loc[df.gene == gene, col] = df.loc[df.gene == gene, col] / self.scale
        return df

    def denormalize_predictions(self, df: pd.DataFrame):
        df = super().denormalize_predictions(df)
        for gene in df['gene'].unique():
            params = self.params.loc[gene].values
            for col in ['predicted_lfc', 'predicted_pm_lfc']:
                if col in df.columns:
                    df.loc[df.gene == gene, col] = df.loc[df.gene == gene, col] * self.scale
                    df.loc[df.gene == gene, col] = self.gaussian_process(df.loc[df.gene == gene, col], self.y, params)
        return df
