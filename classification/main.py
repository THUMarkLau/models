import numpy as np
import preprocess
import decision_tree
import time

src_filename = "train_set.csv"
train_filename = "train.csv"
test_filename = "test.csv"
discrete_keys = ("job", "education", "default", "housing", "loan")
# discrete_keys = ("Alt", "Bar", "Fri", "Hun", "Pat", "Price", "Rain", "Res", "Type", "Est")
continuous_keys = ("age", "balance", "day", "duration", "campaign", "pdays", "previous")


def train():
    data_frame, data_discrete_info, data_continuous_info = preprocess.read_data(train_filename, discrete_keys, [])
    test_frame, _, __ = preprocess.read_data(test_filename, discrete_keys, [])
    # attributes = discrete_keys + continuous_keys
    tree = decision_tree.DecisionTree(data_frame, discrete_keys, data_discrete_info, data_continuous_info, 'y')
    tree.build()
    # tree.show_tree()
    error_rate = tree.inference(test_frame)
    return error_rate


def main(iter=100):
    error_rates = []
    for i in range(iter):
        preprocess.cross_validation(src_filename, train_filename, test_filename, 0.05)
        error_rates.append(train())
    print("Average error rate:", sum(error_rates) / len(error_rates) * 100)
    print("Max error rate:", max(error_rates))
    print("Min error rate:", min(error_rates))


if __name__ == "__main__":
    main(10)
