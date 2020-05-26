import pandas as pd
from sklearn import feature_selection
import numpy as np
import time
import random


def variance_selection(data: pd.DataFrame, threshold=None):
    # 使用方差来对特征进行筛选
    selector = feature_selection.VarianceThreshold()
    selector.fit(data)
    if threshold is None:
        mean_variance = np.mean(selector.variances_)
        selector.threshold = mean_variance
    return data.columns[selector.get_support(True)]


def select_k_best(data: pd.DataFrame, labels, k):
    selector = feature_selection.SelectKBest(score_func=feature_selection.chi2, k=k)
    norm_data = (data - data.min()) / (data.max() - data.min())
    selector.fit(norm_data.values, labels.values)
    return data.columns[selector.get_support(True)]


def read_data(filename, selection_func="variance_selection", k=5):
    # 读取数据，并选择合适的特征
    examples = pd.read_csv(filename)
    features = examples.drop(["Family", "Genus", "Species", "RecordID"], axis=1)
    selected_columns = features.columns
    ground_truth = examples['Species']
    if selection_func == "variance":
        selected_columns = variance_selection(features)
    elif selection_func == "select_k":
        selected_columns = select_k_best(features, ground_truth, k)

    features = examples[selected_columns]
    return features, ground_truth


def generate_smaller_file(org_filename, target_filename, size, keys):
    # 构建一个有size行的文件
    data = pd.read_csv(org_filename)
    random.seed(time.time())
    random_lines = random.sample(range(0, len(data)), size)
    result = data.iloc[random_lines]
    result.to_csv(target_filename, index=False, sep=",")
