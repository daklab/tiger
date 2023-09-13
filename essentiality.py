import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import betaln, digamma, gammaln
from scipy.stats import lomax, norm, pearsonr, spearmanr, t
from statsmodels.stats.diagnostic import lilliefors

LFC_COLS = ['lfc_r1', 'lfc_r2', 'lfc_r3']


class MoNEt(object):

    def __init__(self, mu, sigma, a=0.5, b=0.5, c=1.0, d=1.0):

        # prior parameters
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        # non-targeting distribution parameters
        self.mu = mu
        self.sigma = sigma

        # variational parameters
        self.nu_a = None
        self.nu_b = None
        self.nu_c = None
        self.nu_d = None

        # elbo plot
        self.elbo_plot = None

    def support(self, x):
        x = -(x - self.mu)
        return x[x >= 0]

    def expected_log_p_x_active(self, x):
        return digamma(self.nu_c) - np.log(self.nu_d) - x * self.nu_c / self.nu_d

    def expected_log_p_x_inactive(self, x):
        return (np.log(2) - np.log(self.sigma ** 2) - np.log(np.pi) - x ** 2 / self.sigma ** 2) / 2

    def update_q_lambda(self, x, phi_active):
        self.nu_c = self.c + sum(phi_active)
        self.nu_d = self.d + sum(phi_active * x)

    def update_q_pi(self, phi_active, phi_inactive):
        self.nu_a = self.a + sum(phi_active)
        self.nu_b = self.b + sum(phi_inactive)

    def optimal_q_c(self, x):
        phi_active = np.exp(self.expected_log_p_x_active(x) + digamma(self.nu_a) - digamma(self.nu_a + self.nu_b))
        phi_inactive = np.exp(self.expected_log_p_x_inactive(x) + digamma(self.nu_b) - digamma(self.nu_a + self.nu_b))
        z_phi = phi_active + phi_inactive
        phi_active = phi_active / z_phi
        phi_inactive = phi_inactive / z_phi
        assert 0 <= phi_active.min() <= phi_active.max() <= 1
        assert 0 <= phi_inactive.min() <= phi_inactive.max() <= 1
        return phi_active, phi_inactive

    def elbo(self, x, phi_active, phi_inactive):
        # E[1[ci=1] ln p(xi|c=1,lam)]
        elbo = sum(phi_active * self.expected_log_p_x_active(x))

        # E[1[ci=0] ln p(xi|c=0)]
        elbo += sum(phi_inactive * self.expected_log_p_x_inactive(x))

        # E[1[ci=1] ln p(ci=1|pi)]
        elbo += sum(phi_active) * (digamma(self.nu_a) - digamma(self.nu_a + self.nu_b))

        # E[1[ci=0] ln p(ci=0|pi)]
        elbo += sum(phi_inactive) * (digamma(self.nu_b) - digamma(self.nu_a + self.nu_b))

        # -DKL[q(lambda)||p(lambda)]
        elbo -= digamma(self.nu_c) * (self.nu_c - self.c) - gammaln(self.nu_c) + gammaln(self.c)
        elbo -= self.c * (np.log(self.nu_d) - np.log(self.d)) + self.nu_c * (self.d - self.nu_d) / self.nu_d

        # -DKL[q(pi)||p(pi)]
        elbo -= betaln(self.a, self.b) - betaln(self.nu_a, self.nu_b)
        elbo -= (self.nu_a - self.a) * digamma(self.nu_a) + (self.nu_b - self.b) * digamma(self.nu_b)
        elbo -= (self.a - self.nu_a + self.b - self.nu_b) * digamma(self.nu_a + self.nu_b)

        # H[q(c)]
        elbo += -sum(np.log(phi_active ** phi_active) + np.log(phi_inactive ** phi_inactive))

        return elbo

    def fit(self, x, max_iterations=500):
        x = self.support(x)
        if len(x) > 0:
            self.nu_a, self.nu_b, self.nu_c, self.nu_d = self.a, self.b, self.c, self.d
            self.elbo_plot = np.empty(max_iterations)
            for t in range(max_iterations):
                phi_active, phi_inactive = self.optimal_q_c(x)
                self.update_q_pi(phi_active, phi_inactive)
                self.update_q_lambda(x, phi_active)
                self.elbo_plot[t] = self.elbo(x, phi_active, phi_inactive)
            assert min(np.diff(self.elbo_plot) > -1e-9)

    def plot_results(self, x, fig_title=''):
        x = self.support(x)
        if len(x) > 0:
            fig, ax = plt.subplots()
            fig.suptitle('MoNET: ' + fig_title)
            x_sweep = np.linspace(0, max(x), num=1000)
            pi = self.nu_a / (self.nu_a + self.nu_b)
            prob_x_active = lomax.pdf(x_sweep, self.nu_c, scale=self.nu_b)
            prob_x_inactive = 2 * norm.pdf(x_sweep, loc=0, scale=self.sigma)
            ax.plot(x_sweep, pi * prob_x_active + (1 - pi) * prob_x_inactive, label='PP')
            ax.plot(x_sweep, pi * prob_x_active, label='active')
            ax.plot(x_sweep, (1 - pi) * prob_x_inactive, label='inactive')
            sns.histplot(x=x, stat='density', ax=ax, alpha=0.2)
            return fig


class MoG(object):

    def __init__(self, global_pi, mu_nt, sigma_nt, a_pi=0.5, b_pi=0.5, a_tau=1.0, b_tau=1.0):

        # configuration
        self.global_pi = global_pi

        # prior parameters
        self.a_pi = a_pi
        self.b_pi = b_pi
        self.mu_mu = norm(loc=mu_nt, scale=sigma_nt).ppf(0.05)
        self.tau_mu = sigma_nt ** -2
        self.a_tau = a_tau
        self.b_tau = b_tau

        # non-targeting distribution parameters
        self.mu_nt = mu_nt
        self.sigma_nt = sigma_nt

        # variational parameters
        self.nu_a_pi = None
        self.nu_b_pi = None
        self.nu_mu_mu = None
        self.nu_tau_mu = None
        self.nu_a_tau = None
        self.nu_b_tau = None

        # elbo monitor
        self.elbo_monitor = None

    def expected_log_p_x_c_active(self, i, x):
        return (digamma(self.nu_a_tau[i]) - np.log(self.nu_b_tau[i]) - np.log(2 * np.pi)) / 2 - \
               ((x - self.nu_mu_mu[i]) ** 2 * self.nu_a_tau[i] / self.nu_b_tau[i] + 1 / self.nu_tau_mu[i]) / 2

    def expected_log_p_c_active_pi(self, i):
        if self.global_pi:
            i = 0
        return digamma(self.nu_a_pi[i]) - digamma(self.nu_a_pi[i] + self.nu_b_pi[i])

    def expected_log_p_x_c_inactive(self, x):
        return (-np.log(self.sigma_nt ** 2) - np.log(2 * np.pi) - (x - self.mu_nt) ** 2 / self.sigma_nt ** 2) / 2

    def expected_log_p_c_inactive_pi(self, i):
        if self.global_pi:
            i = 0
        return digamma(self.nu_b_pi[i]) - digamma(self.nu_a_pi[i] + self.nu_b_pi[i])

    def update_q_pi(self, i, phi_active_sum, phi_inactive_sum):
        if self.global_pi:
            i = 0
        self.nu_a_pi[i] = self.a_pi + phi_active_sum
        self.nu_b_pi[i] = self.b_pi + phi_inactive_sum

    def update_q_mu_tau(self, i, x, phi_active):
        self.nu_tau_mu[i] = self.tau_mu + sum(phi_active)
        self.nu_mu_mu[i] = (self.tau_mu * self.mu_mu + sum(phi_active * x)) / self.nu_tau_mu[i]
        self.nu_a_tau[i] = self.a_tau + sum(phi_active) / 2
        self.nu_b_tau[i] = self.b_tau + (self.tau_mu * self.mu_mu ** 2) / 2
        self.nu_b_tau[i] += (sum(phi_active * x ** 2) - self.nu_mu_mu[i] ** 2 * self.nu_tau_mu[i]) / 2

    def optimal_q_c(self, i, x):
        phi_active = np.exp(self.expected_log_p_x_c_active(i, x) + self.expected_log_p_c_active_pi(i))
        phi_inactive = np.exp(self.expected_log_p_x_c_inactive(x) + self.expected_log_p_c_inactive_pi(i))
        z_phi = phi_active + phi_inactive
        phi_active = phi_active / z_phi
        phi_inactive = phi_inactive / z_phi
        assert 0 <= phi_active.min() <= phi_active.max() <= 1
        assert 0 <= phi_inactive.min() <= phi_inactive.max() <= 1
        return phi_active, phi_inactive

    def dkl_pi(self, i):
        if self.global_pi:
            i = 0
        dkl = betaln(self.a_pi, self.b_pi) - betaln(self.nu_a_pi[i], self.nu_b_pi[i])
        dkl += (self.nu_a_pi[i] - self.a_pi) * digamma(self.nu_a_pi[i])
        dkl += (self.nu_b_pi[i] - self.b_pi) * digamma(self.nu_b_pi[i])
        dkl += (self.a_pi - self.nu_a_pi[i] + self.b_pi - self.nu_b_pi[i]) * digamma(self.nu_a_pi[i] + self.nu_b_pi[i])

        return dkl

    def local_elbo(self, i, x, phi_active, phi_inactive):
        # E[1[ci=1] (ln p(xi|c=1,mu,tau) + ln p(ci=1|pi))]
        elbo = sum(phi_active * (self.expected_log_p_x_c_active(i, x) + self.expected_log_p_c_active_pi(i)))

        # E[1[ci=0] (ln p(xi|c=0;mu_nt,tau_nt) + ln p(ci=0|pi))]
        elbo += sum(phi_inactive * (self.expected_log_p_x_c_inactive(x) + self.expected_log_p_c_inactive_pi(i)))

        # H[q(c)]
        elbo += -sum(np.log(phi_active ** phi_active) + np.log(phi_inactive ** phi_inactive))

        # -DKL[q(mu,tau)||p(mu,tau)]
        elbo -= self.nu_a_tau[i] / self.nu_b_tau[i] / 2 * (self.mu_mu - self.nu_mu_mu[i]) ** 2 * self.tau_mu
        elbo -= (self.tau_mu / self.nu_tau_mu[i] - np.log(self.tau_mu) + np.log(self.nu_tau_mu[i])) / 2
        elbo -= self.a_tau * (np.log(self.nu_b_tau[i]) - np.log(self.b_tau))
        elbo -= gammaln(self.a_tau) - gammaln(self.nu_a_tau[i])
        elbo -= (self.nu_a_tau[i] - self.a_tau) * digamma(self.nu_a_tau[i])
        elbo -= (self.b_tau - self.nu_b_tau[i]) * self.nu_a_tau[i] / self.nu_b_tau[i]

        return elbo

    def fit(self, gene_values, max_iterations=500):
        if self.global_pi:
            self.nu_a_pi, self.nu_b_pi = np.array([self.a_pi]), np.array([self.b_pi])
        else:
            self.nu_a_pi, self.nu_b_pi = np.ones(len(gene_values)) * self.a_pi, np.ones(len(gene_values)) * self.b_pi
        self.nu_mu_mu, self.nu_tau_mu = np.ones(len(gene_values)) * self.mu_mu, np.ones(len(gene_values)) * self.tau_mu
        self.nu_a_tau, self.nu_b_tau = np.ones(len(gene_values)) * self.a_tau, np.ones(len(gene_values)) * self.b_tau
        self.elbo_monitor = np.zeros(max_iterations)
        for t in range(max_iterations):
            phi_active_sum, phi_inactive_sum = 0, 0
            for i, x in enumerate(gene_values):
                phi_active, phi_inactive = self.optimal_q_c(i, x)
                self.update_q_mu_tau(i, x, phi_active)
                if not self.global_pi:
                    self.update_q_pi(i, sum(phi_active), sum(phi_inactive))
                    self.elbo_monitor[t] -= self.dkl_pi(i)
                self.elbo_monitor[t] += self.local_elbo(i, x, phi_active, phi_inactive)
                phi_active_sum += sum(phi_active)
                phi_inactive_sum += sum(phi_inactive)
            if self.global_pi:
                self.update_q_pi(None, phi_active_sum, phi_inactive_sum)
                self.elbo_monitor[t] -= self.dkl_pi(None)
        assert min(np.diff(self.elbo_monitor) > -1e-9)

    def posterior_predictive_components(self, i):
        p_x_active = t(df=2 * self.nu_a_tau[i],
                       loc=self.nu_mu_mu[i],
                       scale=(self.nu_a_tau[i] / self.nu_b_tau[i] / (1 + self.nu_tau_mu[i] ** -1)) ** -0.5)
        p_x_inactive = norm(loc=self.mu_nt, scale=self.sigma_nt)
        if self.global_pi:
            i = 0
        prob_c_active = self.nu_a_pi[i] / (self.nu_a_pi[i] + self.nu_b_pi[i])
        prob_c_inactive = self.nu_b_pi[i] / (self.nu_a_pi[i] + self.nu_b_pi[i])

        return prob_c_active, p_x_active, prob_c_inactive, p_x_inactive


def plot_results(x, prob_c_active, p_x_active, prob_c_inactive, p_x_inactive, title='', ax=None):
    if len(x) > 0:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.set_title(title)
        x_min = mixture_quantile(0.001, prob_c_active, p_x_active, prob_c_inactive, p_x_inactive)
        x_max = mixture_quantile(0.999, prob_c_active, p_x_active, prob_c_inactive, p_x_inactive)
        x_sweep = np.linspace(x_min, x_max, num=1000)
        p_x_active = p_x_active.pdf(x_sweep)
        p_x_inactive = p_x_inactive.pdf(x_sweep)
        ax.plot(x_sweep, prob_c_active * p_x_active + prob_c_inactive * p_x_inactive, label='PP')
        ax.plot(x_sweep, prob_c_active * p_x_active, label='active')
        ax.plot(x_sweep, prob_c_inactive * p_x_inactive, label='inactive')
        sns.histplot(x=x, stat='density', ax=ax, alpha=0.2)
        return fig


def mixture_cdf(x, prob_c_active, p_x_active, prob_c_inactive, p_x_inactive):
    return prob_c_active * p_x_active.cdf(x) + prob_c_inactive * p_x_inactive.cdf(x)


def mixture_quantile(p, prob_c_active, p_x_active, prob_c_inactive, p_x_inactive, bits=32):

    # initialize binary search range
    search_range = [min(p_x_active.ppf(p), p_x_inactive.ppf(p)),
                    max(p_x_active.ppf(p), p_x_inactive.ppf(p))]

    # binary search up to specified bits of precision
    x = sum(search_range) / 2
    for _ in range(bits):
        search_range[int(mixture_cdf(x, prob_c_active, p_x_active, prob_c_inactive, p_x_inactive) > p)] = x
        new_x = sum(search_range) / 2
        if new_x == x:
            break
        x = new_x

    return x


def compute_scores(replace):
    os.makedirs('essentiality', exist_ok=True)

    # loop over the datasets
    dataset_params = [
        # ('off-target', {'global_pi': True}),
        ('off-target', {'global_pi': False}),
        # ('junction', {'global_pi': True}),
        ('junction', {'global_pi': False}),
    ]
    for (dataset, params) in dataset_params:
        param_str = '-'.join(key + '-' + str(item) for key, item in params.items())
        figure_path = os.path.join('figures', 'essentiality', dataset, param_str)
        os.makedirs(figure_path, exist_ok=True)

        # only compute gene quantiles if we are forced to do so or if they don't already exist
        quantile_file = os.path.join('essentiality', dataset + '-' + param_str + '.pkl')
        if replace or not os.path.exists(quantile_file):

            # initialize candidate scores container
            df_scores = pd.DataFrame()

            # load non-targeting data
            data_nt = pd.read_pickle(os.path.join('data-processed', dataset + '-nt.pkl'))
            lfc_nt = data_nt[LFC_COLS].values
            lfc_nt = np.reshape(lfc_nt[~np.isnan(lfc_nt)], -1)

            # non-targeting distribution parameters
            mu_non_targeting = np.mean(lfc_nt)
            sigma_non_targeting = np.std(lfc_nt)

            # test if non-targeting LFCs are normally distributed
            _, p_val = lilliefors(lfc_nt)
            print('Lilliefors p-value of non-targeting LFCs: {:.4e}'.format(p_val))
            print('Location: {:.4e} | Scale: {:.4e}'.format(mu_non_targeting, sigma_non_targeting))

            # load targeting data
            data = pd.read_pickle(os.path.join('data-processed', dataset + '.pkl'))

            # bundle LFCs for each gene
            genes = []
            gene_values = []
            for gene in data['gene'].unique():
                lfc = data[(data['gene'] == gene) & (data['guide_type'] == 'PM')][LFC_COLS].values
                lfc = np.reshape(lfc[~np.isnan(lfc)], -1)
                genes.extend([gene])
                gene_values.extend([lfc])

            # fit mixture of a normal and the non-targeting normal
            mog = MoG(mu_nt=mu_non_targeting, sigma_nt=sigma_non_targeting, **params)
            mog.fit(gene_values)

            # loop over the genes
            for i, (gene, values) in enumerate(zip(genes, gene_values)):

                # plot each gene's posterior predictive
                ppc = mog.posterior_predictive_components(i)
                fig = plot_results(values, *ppc, title=gene)
                fig.savefig(os.path.join(figure_path, gene + '.png'))
                plt.close(fig)

                # save each gene's posterior predictive quantiles
                gene_quantiles = dict()
                for percentile in np.arange(5, 100, 5):
                    q = mixture_quantile(percentile / 100, *ppc)
                    gene_quantiles.update({'q' + str(percentile): q})
                df_scores = pd.concat([df_scores, pd.DataFrame(gene_quantiles, index=[gene])])

            # save scores
            df_scores.to_pickle(quantile_file)


def compare_to_rnai(quantile_file):

    # compare screens
    df_cas13 = pd.read_pickle(os.path.join('essentiality', quantile_file))
    df_rnai = pd.read_csv(os.path.join('meta-data', 'D2_combined_gene_dep_scores.csv'))[['Unnamed: 0', 'A375_SKIN']]
    df_rnai.rename(columns={'Unnamed: 0': 'gene'}, inplace=True)
    df_rnai['gene'] = df_rnai['gene'].apply(lambda s: s.split(' ')[0])
    df_compare = df_cas13.join(df_rnai.set_index('gene'), how='inner', lsuffix='_junc', rsuffix='_ot')

    # compute correlation between each screen's fit quantiles
    pearson_values = np.empty(df_cas13.columns.shape)
    spearman_values = np.empty(df_cas13.columns.shape)
    for i, column in enumerate(df_cas13.columns):
        pearson_values[i] = pearsonr(df_compare[column].values, df_compare['A375_SKIN'].values)[0]
        spearman_values[i] = spearmanr(df_compare[column].values, df_compare['A375_SKIN'].values)[0]

    # plot correlations
    fig, ax = plt.subplots()
    fig.suptitle(' '.join(['Comparing'] + quantile_file.split('-') + ['quantiles and RNAi']))
    ax.plot([int(s[1:]) for s in df_cas13.columns], pearson_values, label='Pearson')
    ax.plot([int(s[1:]) for s in df_cas13.columns], spearman_values, label='Spearman')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Correlation')
    ax.legend()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--replace', action='store_true', default=False, help='recompute scores even if they exist')
    args = parser.parse_args()

    # compute scores
    compute_scores(args.replace)

    # compare to RNAi
    compare_to_rnai('junction-global_pi-False.pkl')
    # compare_to_rnai('junction-global_pi-True.pkl')
    compare_to_rnai('off-target-global_pi-False.pkl')
    # compare_to_rnai('off-target-global_pi-True.pkl')

    plt.show()
