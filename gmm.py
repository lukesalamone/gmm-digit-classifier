import pickle
import numpy as np
from kmeans import KMeans
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, n_clusters, covariance_type):
        self.n_clusters = n_clusters
        allowed_covariance_types = ['spherical', 'diagonal']
        if covariance_type not in allowed_covariance_types:
            raise ValueError(f'covariance_type must be in {allowed_covariance_types}')
        self.covariance_type = covariance_type

        self.means = None
        self.covariances = None
        self.mixing_weights = None
        self.max_iterations = 200

    def loadMeans(self, path):
        self.means = np.load(path)

    def fit(self, features):
        if self.means is None:
            kmeans = KMeans(self.n_clusters)
            kmeans.fit(features)
            print('finished initial clustering')
            self.means = kmeans.means
            np.save('means.npy', self.means)

        self.covariances = self._init_covariance(features.shape[-1])
        self.mixing_weights = np.random.rand(self.n_clusters)
        self.mixing_weights /= np.sum(self.mixing_weights)
        prev_log_likelihood = -float('inf')
        log_likelihood = self._overall_log_likelihood(features)

        n_iter = 0
        rounds = 0
        while log_likelihood - prev_log_likelihood > 1e-4 and n_iter < self.max_iterations:
            print('gmm iteration: ', rounds)
            rounds += 1

            prev_log_likelihood = log_likelihood

            assignments = self._e_step(features)
            self.means, self.covariances, self.mixing_weights = (
                self._m_step(features, assignments)
            )

            log_likelihood = self._overall_log_likelihood(features)
            n_iter += 1

    def predict(self, features):
        posteriors = self._e_step(features)
        return np.argmax(posteriors, axis=-1)

    # expectation step
    def _e_step(self, features):
        return np.array([self._posterior(features, k)[0] for k in range(self.n_clusters)]).transpose()

    # maximization step
    def _m_step(self, features, assignments):
        mixing_weights = []
        means = []
        covariances = []
        for cluster in range(self.n_clusters):
            gammas = assignments[:,cluster]
            r = np.sum(gammas)

            # calculate weights
            w = r / len(features)
            mixing_weights.append(w)

            # calculate means
            mu = (gammas @ features) / r
            means.append(mu)

            # calculate covariances
            diff_sq = (features-mu) ** 2
            cov = (gammas @ diff_sq) / r
            covariances.append(cov)

        return np.array(means), np.array(covariances), np.array(mixing_weights)

    def _init_covariance(self, n_features):
        if self.covariance_type == 'spherical':
            return np.random.rand(self.n_clusters)
        elif self.covariance_type == 'diagonal':
            return np.random.rand(self.n_clusters, n_features)

    def _log_likelihood(self, features, k_idx):
        return np.array([np.log(self.mixing_weights[k_idx]) + p for p in multivariate_normal.logpdf(features, self.means[k_idx], self.covariances[k_idx])])

    def _overall_log_likelihood(self, features):
        denom = [
            self._log_likelihood(features, j) for j in range(self.n_clusters)
        ]
        return np.sum(denom)

    def _posterior(self, features, k):
        num = self._log_likelihood(features, k)
        denom = np.array([
            self._log_likelihood(features, j)
            for j in range(self.n_clusters)
        ])

        max_value = denom.max(axis=0, keepdims=True)
        denom_sum = max_value + np.log(np.sum(np.exp(denom - max_value), axis=0))
        posteriors = np.exp(num - denom_sum)
        return posteriors
