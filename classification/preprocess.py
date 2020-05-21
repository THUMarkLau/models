import pandas
import random
import time
from sklearn.utils import shuffle


def read_data(filename, discrete_keys, continuous_keys):
    # 返回data frame，以及在每个离散值key的所有可能对应的value
    # 返回每个连续值key所对应的value的最大最小值
    data = pandas.read_csv(filename)
    discrete_values = {}
    continuous_values = {}
    for key in discrete_keys:
        value = set(data[key])
        discrete_values[key] = value

    for key in continuous_keys:
        value = {'max': data[key].max(), 'min': data[key].min(), 'mean': data[key].mean()}
        continuous_values[key] = value

    return data, discrete_values, continuous_values


def generate_smaller_file(org_filename, target_filename, size, keys):
    # 构建一个有size行的文件
    data = pandas.read_csv(org_filename)
    result = data[list(keys)]
    random.seed(time.time())
    random_lines = random.sample(range(0, len(data)), size)
    result = result.iloc[random_lines]
    result.to_csv(target_filename, index=False, sep=",")


def cross_validation(src_filename, train_filename, test_filename, test_percentage):
    all_data = pandas.read_csv(src_filename)
    all_data_count = len(all_data)
    all_data = shuffle(all_data)
    test_size = int(test_percentage * all_data_count)
    test_data = all_data[:test_size]
    train_data = all_data[test_size:]
    test_data.to_csv(test_filename, index=False, sep=",")
    train_data.to_csv(train_filename, index=False, sep=",")
