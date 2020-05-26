运行方式 python main.py
参数：
    --distance 距离参数，可选则 euclid 或 manhattan，默认为manhattan
    --filename 文件名
    --feature 特征提取算法，可选择 variance 或 select_k，默认为 select_k
    --show 是否可视化，可选择 y 或 n，默认为n
使用参数运行 python main.py --distance manhattan --filename train.csv --feature select_k --show y

注意，train.csv是一个规模为50的数据集，从train_set.csv中随机采样生成。如果想要可视化，建议使用该文件。