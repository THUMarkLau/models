import numpy as np
import preprocess
import decision_tree
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--feature", type=int, default=2, choices=[1, 2])
args = parser.parse_args()
src_filename = "train_set.csv"
train_filename = "train.csv"
test_filename = "test.csv"
if args.feature == 1:
    discrete_keys = ("job", "education", "default", "housing", "loan")
    continuous_keys = ("age", "balance")
elif args.feature == 2:
    discrete_keys = ("contact", "month", "poutcome")
    continuous_keys = ("day", "duration", "campaign", "pdays", "previous")
else:
    print("feature must be 1 or 2")
    exit(0)

def dt():
    start_time = time.time()
    data_frame, data_discrete_info, data_continuous_info = preprocess.read_data(train_filename, discrete_keys,
                                                                                continuous_keys)
    test_frame, _, __ = preprocess.read_data(test_filename, discrete_keys, continuous_keys)
    # attributes = discrete_keys + continuous_keys
    tree = decision_tree.DecisionTree(data_frame, discrete_keys + continuous_keys, data_discrete_info,
                                      data_continuous_info, 'y')
    tree.build()
    # tree.show_tree()
    error_rate = tree.inference(test_frame)
    end_time = time.time()
    print("Time cost:", end_time - start_time)
    return error_rate


def _decision_tree(iter=100):
    error_rates = []
    for i in range(iter):
        # preprocess.cross_validation(src_filename, train_filename, test_filename, 0.05)
        error_rates.append(train())
    return 1 - error_rates


def bayes_data(test_percentage=0.1):
    # 将数据处理为朴素贝叶斯可以处理的数据
    data, discrete_infos, continuous_infos = preprocess.read_data('train_set.csv', discrete_keys, continuous_keys)
    # 将数据进行打乱
    data_len = len(data)
    test_data_count = test_percentage * data_len
    data = shuffle(data)
    ground_truth = data['y']
    features = data[list(discrete_keys + continuous_keys)]
    # 将数据处理为数值数据，并且通过归一化将数据转换为非负的
    for discrete_key in discrete_keys:
        idx = 0
        for val in discrete_infos[discrete_key]:
            features = features.replace(to_replace=val, value=idx)
            idx += 1
    features = (features - features.min()) / (features.max() - features.min())


    return features, ground_truth


def bayes():
    train_features, train_labels = bayes_data()
    model = MultinomialNB()
    # 训练
    return cross_val_score(model, train_features, train_labels)


def mlp_data():
    # 将数据预处理成可以被mlp使用的数据形式

    # 打乱数据
    preprocess.cross_validation("train_set.csv", "train.csv", "test.csv", 0.1)
    # 读取数据
    train_data, discrete_values, continous_values = preprocess.read_data("train_set.csv", discrete_keys, continuous_keys)
    # 将离散数据转换成连续数据
    for key in discrete_values.keys():
        idx = 0
        for val in discrete_values[key]:
            train_data = train_data.replace(to_replace=val, value=idx)

    train_features = train_data[list(discrete_keys + continuous_keys)]
    train_ground_truth = train_data['y']

    return train_features, train_ground_truth


def mlp():
    model = MLPClassifier(
        solver='adam',
        activation='relu',
        max_iter=10,
        alpha=1e-5,
        hidden_layer_sizes=(len(discrete_keys + continuous_keys), 20, 1),
        random_state=1,
        verbose=False
    )
    train_features, train_result = mlp_data()
    return cross_val_score(model, train_features, train_result)


if __name__ == "__main__":
    time1 = time.time()
    print("-"*50)
    print("Decision tree running...")
    error_rate = dt()
    time2 = time.time()
    print("Decision tree running finish")
    print("Error rate", error_rate)
    print("Time consume(s): ", time2-time1)
    print("-"*50)
    print("Bayes running...")
    time1 = time.time()
    score = bayes()
    time2 = time.time()
    print("Bayes running finish")
    print("Error rate", 1 - sum(score) / len(score))
    print("Time consume(s): ", time2-time1)
    print("-"*50)
    print("MLP running...")
    time1 = time.time()
    score = mlp()
    time2 = time.time()
    print("MLP running finish")
    print("Error rate", 1 - sum(score) / len(score))
    print("Time consumne(s): ", time2-time1)
    print("-"*50)
