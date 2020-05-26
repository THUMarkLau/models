import pandas as pd
import math
import pdb
import copy

show_count = 0


class NodeType:
    INNER_NODE = 0
    LEAF = 1

class AttributeType:
    DISCRETE = 0
    CONTINUOUS = 1

class Attribute:

    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.division_point = -1

    def set_division_point(self, division_point):
        self.division_point = division_point


class DecisionTreeNode:

    def __init__(self):
        global show_count
        self.type = None
        self.output = None
        self.attr = None
        self.forward_map = {}
        self.num = show_count
        show_count += 1

    def set_type(self, type):
        self.type = type

    def set_output(self, output):
        self.output = output

    def set_attr(self, attr):
        self.attr = attr

    def set_sub_node(self, key, sub_node):
        self.forward_map[key] = sub_node

    def forward(self, example):
        if self.type == NodeType.INNER_NODE:
            # 内节点
            if self.attr.type == AttributeType.DISCRETE:
                value = example[self.attr.name]
                next_node = self.forward_map[value]
            elif self.attr.type == AttributeType.CONTINUOUS:
                if example[self.attr.name] > self.attr.division_point:
                    next_node = self.forward_map['greater']
                else:
                    next_node = self.forward_map['smaller']
            return next_node.forward(example)
        else:
            # 叶子节点
            return self.output

    def show(self):
        if self.type == NodeType.LEAF:
            # 叶节点
            print("-" * 50)
            print("Leaf node %d" % self.num, end=' ')
            print("output:", self.output)
        else:
            # 内部节点
            print("-" * 50)
            print("Inner node %d" % self.num)
            print("Attr:", self.attr)
            for value in self.forward_map.keys():
                print("Value", value, "-> Node", self.forward_map[value].num)
            for value in self.forward_map.keys():
                self.forward_map[value].show()


class DecisionTree:

    def __init__(self, data, attributes, discrete_attr_info, continuous_attr_info, result_key):
        self.data = data
        self.attributes = []
        for attribute in attributes:
            if attribute in discrete_attr_info.keys():
                self.attributes.append(Attribute(attribute, AttributeType.DISCRETE))
            elif attribute in continuous_attr_info.keys():
                self.attributes.append(Attribute(attribute, AttributeType.CONTINUOUS))
        self.discrete_attr_info = discrete_attr_info
        self.continuous_attr_info = continuous_attr_info
        self.result_key = result_key
        self.root_node = None

    def _get_result(self, data):
        # 获取某一行数据的结果
        return data[self.result_key]

    def _is_same_kind(self, examples):
        # 判断是否所有的样本的类别都是相同的
        result_set = set(examples[self.result_key])
        if len(result_set) == 1:
            return True
        else:
            return False

    def _is_same_attr_value(self, examples, attribute):
        # 判断是否所有数据都在同一个属性上取相同的值
        if len(set(examples[attribute.name])) == 1:
            return True
        else:
            return False

    def _is_same_attrs_value(self, data, attributes):
        # 判断是否所有data在attribute上的取值都一样
        for attribute in attributes:
            if not self._is_same_attr_value(data, attribute):
                return False
        return True

    def _most_attributes(self, data):
        # 返回 data 中样本数最多的类
        positive_count = len(data[data[self.result_key] == 1])
        negative_count = len(data[data[self.result_key] == 0])
        if positive_count >= negative_count:
            return 1
        else:
            return 0

    def _entropy(self, examples):
        # 计算样本中的熵
        result = 0
        positive_count = len(examples[examples[self.result_key] == 1])
        negative_count = len(examples[examples[self.result_key] == 0])
        total_count = len(examples)
        if total_count == 0 or positive_count == 0 or negative_count == 0:
            return 0
        positive_percentage = positive_count / total_count
        negative_percentage = negative_count / total_count
        result -= positive_percentage * math.log2(positive_percentage)
        result -= negative_percentage * math.log2(negative_percentage)
        return result

    def _info_gain_discrete_attribute(self, entropy_origin, size_origin, partitions):
        # 返回经过划分之后减少的熵
        gain = entropy_origin
        for partition in partitions:
            partition_size = len(partition)
            gain -= (partition_size / size_origin) * self._entropy(partition)
        return gain

    def _intrinsic_value(self, size_origin, partitions):
        # 返回某个划分的固有值
        intrinsic_value = 0
        for partition in partitions:
            partition_size = len(partition)
            if partition_size == 0:
                continue
            intrinsic_value -= (partition_size / size_origin) * math.log2(partition_size / size_origin)
        if intrinsic_value == 0:
            intrinsic_value = 0.001
        return intrinsic_value

    def _info_gain_ratio(self, info_gain, size_origin, partitions):
        # 返回经过划分之后的增益率
        intrinsic_value = self._intrinsic_value(size_origin, partitions)
        return info_gain / intrinsic_value

    def _get_attr_all_possible_values(self, attribute):
        # 返回某个属性的所有可能取值
        if attribute.type == AttributeType.DISCRETE:
            return self.discrete_attr_info[attribute.name]
        else:
            return ['greater', 'smaller']

    def _get_discrete_attribute_partition(self, data, attribute, values):
        # 将数据以 attribute 的不同取值划分为不同的集合
        result = {}
        for value in values:
            part = data[data[attribute.name] == value]
            result[value] = part
        return result


    def _get_division_points(self, attribute, examples):
        # 获取连续数据的分割点集合
        # 获取数据域上的所有值，并且进行排序
        values = list(set(examples[attribute.name]))
        values.sort()

        division_points = []
        for i in range(len(values)-1):
            division_points.append((values[i] + values[i+1]) / 2)

        return division_points

    def _get_continuous_attribute_partition(self, attribute, examples, division_point):
        partition = {}
        greater_part = examples[examples[attribute.name] > division_point]
        smaller_part = examples[examples[attribute.name] <= division_point]
        partition['greater'] = greater_part
        partition['smaller'] = smaller_part
        return partition

    def _info_gain_continuous(self, attribute, examples):
        # 获取连续值的最大信息增益及其划分点
        entropy_ori = self._entropy(examples)
        size_origin = len(examples)
        division_points = self._get_division_points(attribute, examples)
        max_info_gain = -1
        optimal_division_point = -1
        for division_point in division_points:
            partitions = self._get_continuous_attribute_partition(attribute, examples, division_point)
            info_gain = entropy_ori
            for partition in partitions.values():
                info_gain -= len(partition) / size_origin * self._entropy(partition)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                optimal_division_point = division_point
        return max_info_gain, optimal_division_point


    def _optimal_attribute(self, data, attributes):
        # 返回一个最优属性划分
        entropy_origin = self._entropy(data)
        size_origin = len(data)

        info_gains = {}
        info_gains_ratio = {}

        # 计算每一种属性的信息增益和增益率
        for attribute in attributes:
            if attribute.type == AttributeType.DISCRETE:
                # 离散值
                partition = self._get_discrete_attribute_partition(
                    data,
                    attribute,
                    self._get_attr_all_possible_values(attribute)
                ).values()
                info_gains[attribute.name] = self._info_gain_discrete_attribute(entropy_origin, size_origin, partition)
                info_gains_ratio[attribute.name] = self._info_gain_ratio(info_gains[attribute.name], size_origin, partition)
            elif attribute.type == AttributeType.CONTINUOUS:
                # 连续值
                info_gain, division_point = self._info_gain_continuous(attribute, data)
                attribute.set_division_point(division_point)
                info_gains[attribute.name] = info_gain
                partition = self._get_continuous_attribute_partition(attribute, data, division_point).values()
                info_gains_ratio[attribute.name] = self._info_gain_ratio(info_gain, size_origin, partition)

        avg_info_gains = sum(info_gains.values()) / len(info_gains)
        optimal_attribute = None
        max_info_gain_ratio = -1
        for attribute in attributes:
            if info_gains[attribute.name] >= avg_info_gains:
                if info_gains_ratio[attribute.name] >= max_info_gain_ratio:
                    optimal_attribute = attribute
                    max_info_gain_ratio = info_gains_ratio[attribute.name]

        return optimal_attribute

    # def _optimal_attribute(self, data, attributes):
    #     # 返回一个最优属性划分
    #     entropy_origin = self._entropy(data)
    #     size_origin = len(data)
    #     max_info_gain = -1
    #     optimal_attribute = None
    #     partition_count = 1000
    #
    #     # 计算每一种属性的信息增益和增益率
    #     for attribute in attributes:
    #         partition = self._get_attribute_partition(
    #             data,
    #             attribute,
    #             self._get_all_possible_values(attribute)
    #         ).values()
    #         info_gain = round(self._info_gain(entropy_origin, size_origin, partition), 4)
    #         if info_gain > max_info_gain:
    #             max_info_gain = info_gain
    #             optimal_attribute = attribute
    #             partition_count = len(partition)
    #         elif info_gain == max_info_gain:
    #             # 相同的情况下选择数目比较少的那种
    #             if len(partition) < partition_count:
    #                 optimal_attribute = attribute
    #                 partition_count = len(partition)
    #
    #     return optimal_attribute

    def _remove_attr_from_list(self, attributes, attr):
        for attribute in attributes:
            if attribute.name == attr.name:
                attributes.remove(attribute)
                return

    def _get_attribute_partition(self, examples, attribute, possible_values):
        if attribute.type == AttributeType.DISCRETE:
            return self._get_discrete_attribute_partition(examples, attribute, possible_values)
        else:
            return self._get_continuous_attribute_partition(attribute, examples, attribute.division_point)

    def _tree_generate(self, data, attributes: list):
        # 生成一个节点
        node = DecisionTreeNode()
        if self._is_same_kind(data):
            # 如果所有的数据都是同一个类别C
            # 则将节点标记为叶节点，并且输出为类别C
            node.set_type(NodeType.LEAF)
            node.set_output(self._get_result(data.iloc[0]))
            return node
        if len(attributes) == 0 or \
                self._is_same_attrs_value(data, attributes):
            # 如果 attributes 为空或
            # data 在所有 attributes 上的取值都相同
            # 则中止，将该节点标记为叶节点，其分类为样本中最多的类
            node.set_type(NodeType.LEAF)
            node.set_output(self._most_attributes(data))
            return node

        # 寻找最优划分属性 a*
        optimal_attribute = self._optimal_attribute(data, attributes)
        # 最优划分属性的所有可能取值
        possible_values = self._get_attr_all_possible_values(optimal_attribute)
        # 使用最优划分属性对数据进行一个划分
        data_partition = self._get_attribute_partition(data, optimal_attribute, possible_values)
        # 生成分支
        node.set_type(NodeType.INNER_NODE)
        node.set_attr(optimal_attribute)
        new_attributes = copy.deepcopy(attributes)
        self._remove_attr_from_list(new_attributes, optimal_attribute)
        for value in possible_values:
            part = data_partition[value]
            if len(part) == 0:
                # 如果有一个划分是空的
                sub_node = DecisionTreeNode()
                sub_node.set_type(NodeType.LEAF)
                sub_node.set_output(self._most_attributes(data))
            else:
                sub_node = self._tree_generate(part, new_attributes)
            node.set_sub_node(value, sub_node)

        return node

    def build(self):
        self.root_node = self._tree_generate(self.data, list(self.attributes))

    def inference(self, examples):
        # 进行推理
        ground_truth = examples[self.result_key]
        test_result = pd.DataFrame.copy(ground_truth)
        for i in range(len(examples)):
            test_result[i] = self.root_node.forward(examples.iloc[i])
        error_rate = (ground_truth == test_result).value_counts()[False] / len(ground_truth)
        return error_rate

    def show_tree(self):
        self.root_node.show()
