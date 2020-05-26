import k_means
import distance
import preprocess
import time
from sklearn.metrics import jaccard_similarity_score, davies_bouldin_score
from sklearn.cluster import Birch
from sklearn.manifold import TSNE
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--distance", type=str, default="euclid")
parser.add_argument("--filename", type=str, default="sub.csv")
parser.add_argument("--feature", type=str, default="variance")
parser.add_argument("--show", type=str, default='n', choices=['y','n'])
args = parser.parse_args()


def kmeans(features, k, distance_func=distance.euclid_distance):
    clustering = k_means.KMeans(features, k, distance_func)
    labels = clustering.run()
    return labels


if __name__ == '__main__':
    distance_func_name = args.distance
    filename = args.filename
    tsne = TSNE()
    if args.feature != 'variance' and args.feature != 'select_k':
        print("Feature selection function should be variance or select_k")
        exit(0)
    features, ground_truth = preprocess.read_data(filename, args.feature)
    cluster_count = len(set(ground_truth))
    ground_truth_vals = set(ground_truth)
    mapping = {}
    for idx, val in enumerate(ground_truth_vals):
        ground_truth = ground_truth.replace(to_replace=val, value=idx)
    distance_func = distance.euclid_distance
    if distance_func_name == 'euclid':
        pass
    elif distance_func_name == 'manhattan':
        distance_func = distance.manhattan_distance
    print('-' * 50)
    print('KMeans running...')
    label_kmeans = kmeans(features.values, cluster_count, distance_func)
    print('KMeans running finish')
    print('The Jaccard Similarity score of KMeans result: %.3f' % jaccard_similarity_score(ground_truth, label_kmeans))
    print('The Davies Bouldin score of KMeans result: %f' % davies_bouldin_score(features.values, label_kmeans))
    if args.show == 'y':
        show_data = tsne.fit_transform(features.values[:500])
        plt.scatter(show_data[:, 0], show_data[:, 1], c=label_kmeans[:500])
        plt.show()
    print('-' * 50)
    print('Birch running...')
    model = Birch(n_clusters=cluster_count)
    model.fit(features.values)
    label_birch = model.predict(features.values)
    if args.show =='y':
        plt.scatter(show_data[:, 0], show_data[:, 1], c=label_birch[:500])
        plt.show()
    print('Birch running finish')
    print('The Jaccard Similarity score of Birch result: %.3f' % jaccard_similarity_score(ground_truth, label_birch))
    print('The Davies Bouldin score of Birch result: %f' % davies_bouldin_score(features.values, label_birch))
    print('-' * 50)
