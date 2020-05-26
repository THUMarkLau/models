import numpy as np


def minkowski_disatance(x1, x2, p):
    # 闵科夫斯基距离
    diff = np.abs(x1 - x2)
    sum_diff = np.sum(diff ** p)
    return sum_diff ** (1 / p)


def euclid_distance(x1, x2):
    return minkowski_disatance(x1, x2, 2)

def manhattan_distance(x1, x2):
    return minkowski_disatance(x1, x2, 1)