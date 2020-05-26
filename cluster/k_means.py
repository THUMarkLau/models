import numpy as np
import random
from collections import defaultdict


class KMeans:
    def __init__(self, examples, cluster_count, distance_func):
        self.examples = examples
        self.cluster_count = cluster_count
        self.distance_func = distance_func

    def select_random_k(self, examples, k):
        # 随机选取 k 个数据作为原始的中心点
        center_vectors = []
        examples_len = len(examples)
        numbers = [i for i in range(examples_len)]
        number_choice = random.sample(numbers, k)
        for idx in number_choice:
            center_vectors.append(examples[idx])
        return center_vectors

    def clustering(self, examples, centers):
        label = []
        for example in examples:
            nearest = float('inf')
            nearest_idx = 0
            for i in range(len(centers)):
                distance = self.distance_func(example, centers[i])
                if distance < nearest:
                    nearest = distance
                    nearest_idx = i
            label.append(nearest_idx)
        return label

    def get_center(self, examples):
        dims = len(examples[0])

        new_centers = []

        for dim in range(dims):
            dim_sum = 0
            for example in examples:
                dim_sum += example[dim]
            if len(examples) == 0:
                new_centers.append(0)
            else:
                new_centers.append(dim_sum / len(examples))

        return np.array(new_centers)

    def update_center_vectors(self, examples, labels):
        mean = defaultdict(list)
        center_vectors = [0 for i in range(len(set(labels)))]
        for label, example in zip(labels, examples):
            mean[label].append(example)

        for idx in mean.keys():
            center_vectors[idx] = self.get_center(mean[idx])

        return center_vectors

    def run(self):
        k_points = self.select_random_k(self.examples, self.cluster_count)
        labels = self.clustering(self.examples, k_points)
        pre_labels = None
        while labels != pre_labels:
            new_center_vectors = self.update_center_vectors(self.examples, labels)
            pre_labels = labels
            labels = self.clustering(self.examples, new_center_vectors)

        return np.array(labels)