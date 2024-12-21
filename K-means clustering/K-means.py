import numpy as np
import random
import copy
import matplotlib
import matplotlib.pyplot as plt

# 两种确定初始均值向量的方法
# 类中心在最小最大值范围内随机取值
def center_minmax(data, k):
    n, m = np.shape(data)
    cent_list = []
    for i in range(k):
        cent = []
        for j in range(m):
            mi, mx = min(data[:, j]), max(data[:, j])
            cent.append(mi + random.random() * (mx - mi))
        cent_list.append(cent)
    return cent_list

# 类中心在k个样本点中随机选取
def center_rand(data, k):
    # 将数据转换为列表形式
    data_list = data.tolist()
    return random.sample(data_list, k)

# 距离度量函数
def evaDist(arrA, arrB):
    return np.sqrt(np.sum(np.power((arrA - arrB), 2)))

def kmeans(data, k, max_iter):
    n, m = np.shape(data)  # 获取数据的行数（样本数）和列数（特征数）
    center = center_minmax(data, k)  # 使用 center_minmax 初始化类中心
    data_dict = {}  # 存储每个样本的相关信息
    class_dict = {}  # 存储每个簇包含的样本
    for i in range(n):
        data_dict[i] = [None, None, data[i], None]  # 初始化每个样本的字典：所属类别、类中心、数据点、距离
    flag = True  # 用于控制迭代
    itr = 0  # 迭代计数器
    while flag:
        print (f"聚类簇数 = {k}", f"迭代次数 = {itr+1}")
        itr += 1
        flag = False  # 假设聚类已经收敛
        data_dict_old = copy.deepcopy(data_dict)  # 保留上次迭代的结果，检查是否有变化
        for i in range(k):
            class_dict[i] = []  # 初始化每个簇的样本列表
        for i in range(n):
            dist_min = float('inf')  # 用于记录每个样本点到质心的最小距离
            for j in range(k):
                dist = np.sum(np.power(data[i] - center[j], 2)) ** 0.5  # 计算距离
                if dist < dist_min:
                    dist_min = dist  # 更新最小距离
                    data_dict[i][0] = j  # 更新样本点所属簇
                    data_dict[i][1] = center[j]  # 更新该点的均值向量
                    data_dict[i][-1] = dist  # 更新该点到类中心心的距离
            class_dict[data_dict[i][0]].append(data[i])  # 将样本点加入到对应簇的列表中
            # 如果样本点与类中心心的距离发生变化，则继续迭代
            if data_dict[i][-1] != data_dict_old[i][-1]:
                flag = True
        # 更新质心：对每个簇中的所有样本计算平均值
        for key, value in class_dict.items():
            if value:  # 如果该簇不为空
                data_class = np.vstack(value)  # 将该簇的所有数据点堆叠成矩阵
                center[key] = np.mean(data_class, axis=0)  # 更新质心为该簇所有样本的均值
        if itr > max_iter:  # 如果超过最大迭代次数，停止
            break
    # 计算轮廓系数
    s = 0
    for key, value in class_dict.items():
        for val in value:
            a = 0
            for vl in value:
                a += evaDist(val, vl)  # 计算样本点与同簇内其他点的距离
            a /= len(value)  # 计算簇内的平均距离
            blst = []
            for key1, value1 in class_dict.items():
                if key1 == key:
                    continue
                else:
                    b = 0
                    if len(value1) > 0:
                        for val1 in value1:
                            b += evaDist(val, val1)  # 计算样本点与其他簇的平均距离
                        blst.append(b / len(value1))  # 记录该点与其他簇的平均距离
            if blst:  # 确保 blst 非空
                s += (min(blst) - a) / max(min(blst), a)  # 计算轮廓系数
    return s / n, data_dict, center  # 返回轮廓系数、数据字典和类中心列表



def show(data, k, everyDict, center):
    # 定义一些基本的标记符号
    symbols = ['o', 's', '^', 'D', 'v', 'p', '*', '<', '>', 'H']  # 可以有多种形状

    # 使用 matplotlib 的颜色映射来生成颜色
    colormap = matplotlib.colormaps['tab10']  # 'tab10' 是一个常见的 10 种颜色的调色板
    n_colors = len(colormap.colors)

    for i in range(np.shape(data)[0]):
        # 获取每个点所属的簇编号
        cluster_id = everyDict[i][0]
        # 选择对应的符号和颜色
        symbol = symbols[cluster_id % len(symbols)]  # 如果簇大于 symbol 数量，使用取余
        color = colormap.colors[cluster_id % n_colors]   # 确保颜色不会重复
        plt.plot(data[i][0], data[i][1], marker=symbol, color=color, markersize=10)

    # 绘制类中心
    for lei in range(k):
        plt.plot(center[lei][0], center[lei][1], marker='X', color='r', markersize=6)
    plt.show()


if __name__ == "__main__":
    # 定义西瓜数据集，忽略最后一列标签，四个特征分别为：密度、含糖率、色泽、根蒂
    watermelon_data = [
        [0.697, 0.460, 0.8, 0.7],
        [0.774, 0.376, 0.9, 0.8],
        [0.634, 0.264, 0.6, 0.6],
        [0.608, 0.318, 0.7, 0.5],
        [0.556, 0.215, 0.6, 0.6],
        [0.403, 0.237, 0.5, 0.4],
        [0.481, 0.149, 0.6, 0.5],
        [0.437, 0.211, 0.4, 0.3],
        [0.666, 0.091, 0.8, 0.7],
        [0.243, 0.267, 0.3, 0.4],
        [0.245, 0.057, 0.2, 0.3],
        [0.343, 0.099, 0.4, 0.3],
        [0.639, 0.161, 0.7, 0.6],
        [0.657, 0.198, 0.8, 0.7],
        [0.360, 0.370, 0.5, 0.4],
        [0.593, 0.042, 0.7, 0.5],
        [0.719, 0.103, 0.9, 0.8]
    ]

    watermelon_data = np.array(watermelon_data)

    kdict = {}
    for k in range(2, 6):  # 对不同的 k 值进行实验
        sk, data_dict, center = kmeans(watermelon_data, k, 100)  # 运行 K-means 聚类
        kdict[k] = [sk, data_dict, center]  # 保存结果（轮廓系数、数据字典和类中心列表）

    bestk = sorted(kdict.items(), key=lambda x: x[1][0], reverse=True)[0][0]  # 选择最佳 k
    print('最优聚类簇数:', bestk, '\n最优聚类各簇类中心:', kdict[bestk][-1])

    show(watermelon_data, bestk, kdict[bestk][1], kdict[bestk][-1])  # 显示最佳聚类结果


