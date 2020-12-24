import numpy as np
import numpy.random as rand

INFINITY = float('inf')

def distance(a, b):
    return np.linalg.norm(a-b)

class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.means = []

    def fit(self, features):
        features = np.apply_along_axis(lambda f: {'feature': f, 'cluster': -1}, 1, features)
        self.means = list(map(lambda x: features[x]['feature'], rand.choice(len(features), self.n_clusters, False)))

        updated = True
        iteration = 0
        while updated:
            iteration += 1

            updated, features = self.update_assignments(features)
            if not updated:
                return

            self.update_means(features)

    def predict(self, features):
        return np.apply_along_axis(lambda feature: self.closest_cluster_index(feature), 1, features)

    def update_assignments(self, features):
        updated = False

        for f in features:
            i = self.closest_cluster_index(f['feature'])
            if i != f['cluster']:
                updated = True
                f['cluster'] = i

        return updated, features

    def closest_cluster_index(self, feature):
        min_distance = INFINITY
        min_index = 0

        for i in range(self.n_clusters):
            d = distance(feature, self.means[i])
            if d < min_distance:
                min_distance = d
                min_index = i

        return min_index

    # update means to be averages of each feature in that cluster
    def update_means(self, features):
        means = [[] for i in range(self.n_clusters)]

        for f in features:
            means[f['cluster']].append(f['feature'])

        self.means = list(map(lambda cluster: np.mean(cluster, 0), means))
