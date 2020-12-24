from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
import statistics

def accuracy(predicted, actual):
    matches = 0
    for p,a in zip(predicted, actual):
        matches = matches + 1 if p == a else matches

    return matches / len(predicted)

# return ami and accuracy
def get_stats(predictions, actual):
    ami = adjusted_mutual_info_score(predictions, actual)

    clusters = {}
    for i in range(len(predictions)):
        predicted_cluster = predictions[i]
        actual_label = actual[i]
        if predicted_cluster in clusters:
            clusters[predicted_cluster].append(actual_label)
        else:
            clusters[predicted_cluster] = [actual_label]

    # map from cluster to its label
    cluster_labels = {}
    for cluster in clusters:
        cluster_labels[cluster] = statistics.mode(clusters[cluster])

    predicted_labels = list(map(lambda x: cluster_labels[x], predictions))
    acc = accuracy(predicted_labels, actual)
    return ami, acc


# find the mean of all examples belonging to a cluster
def find_means_neighbors(preds, images):
    clusters = {}
    for i in range(len(preds)):
        predicted_cluster = preds[i]
        image = images[i]

        if predicted_cluster in clusters:
            clusters[predicted_cluster].append(image)
        else:
            clusters[predicted_cluster] = [image]

    means = [None] * 10
    for c in clusters:
        means[c] = np.mean(clusters[c], 0).reshape((28, 28))

    nns = [None] * 10
    for c in clusters:
        mean_feature = means[c]
        selected_feature = None
        min_dist = float('inf')
        for f in clusters[c]:
            dist = np.linalg.norm(mean_feature - f.reshape((28, 28)))
            if dist < min_dist:
                min_dist = dist
                selected_feature = f
        nns[c] = selected_feature

    return means, nns