import pandas as pd
import numpy as np
import math

# 数据集
data = pd.DataFrame({
    '色泽': ['青绿', '乌黑', '乌黑', '青绿', '浅白', '青绿', '乌黑', '乌黑', '乌黑', '青绿', '浅白', '浅白', '青绿',
             '浅白', '乌黑', '浅白', '青绿'],
    '根蒂': ['蜷缩', '蜷缩', '蜷缩', '蜷缩', '蜷缩', '稍蜷', '稍蜷', '稍蜷', '稍蜷', '硬挺', '硬挺', '蜷缩', '稍蜷',
             '稍蜷', '稍蜷', '蜷缩', '蜷缩'],
    '敲声': ['浊响', '沉闷', '浊响', '沉闷', '浊响', '浊响', '浊响', '浊响', '沉闷', '清脆', '清脆', '浊响', '浊响',
             '沉闷', '浊响', '浊响', '沉闷'],
    '纹理': ['清晰', '清晰', '清晰', '清晰', '清晰', '清晰', '稍糊', '清晰', '稍糊', '清晰', '模糊', '模糊', '稍糊',
             '稍糊', '清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '凹陷', '凹陷', '凹陷', '凹陷', '稍凹', '稍凹', '稍凹', '稍凹', '平坦', '平坦', '平坦', '凹陷',
             '凹陷', '稍凹', '平坦', '稍凹'],
    '触感': ['硬滑', '硬滑', '硬滑', '硬滑', '硬滑', '软粘', '软粘', '硬滑', '硬滑', '软粘', '硬滑', '软粘', '硬滑',
             '硬滑', '软粘', '硬滑', '硬滑'],
    '密度': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360,
             0.593, 0.719],
    '含糖率': [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370,
               0.042, 0.103],
    '质量': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
})

# 目标变量
target = '质量'


# 计算准确率的函数
def accuracy(predictions, actual):
    return np.mean(predictions == actual)


# 计算离散特征分类后的准确率
def accuracy_discrete(data, feature, target):
    correct_predictions = 0
    total = len(data)
    for value in data[feature].unique():
        # 对每个类别，选择该类别中数量最多的标签作为预测值
        subset = data[data[feature] == value]
        majority_class = subset[target].mode()[0]  # 选择出现次数最多的标签
        correct_predictions += (subset[target] == majority_class).sum()  # 统计正确分类的数量
    #print(correct_predictions / total)
    return correct_predictions / total


# 计算连续特征分类后的准确率（使用对率回归）
def accuracy_continuous(data, feature, target):
    X = data[feature].values.reshape(-1, 1)
    y = data[target]

    # 用对率回归计算
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)

    # 预测并计算准确率
    y_pred = model.predict(X)
    #print(accuracy(y_pred, y))
    return accuracy(y_pred, y)


# 选择最佳特征
def best_feature(data, target):
    features = data.columns.drop(target)
    max_accuracy = -1
    best_feat = None

    for feature in features:
        if data[feature].dtype == 'object':  # 离散值
            acc = accuracy_discrete(data, feature, target)
        else:  # 连续值
            acc = accuracy_continuous(data, feature, target)

        if acc > max_accuracy:
            max_accuracy = acc
            best_feat = feature

    return best_feat


# 构建决策树
def create_tree(data, target):
    labels = data[target]
    if len(labels.unique()) == 1:  # 如果所有标签相同，返回叶节点
        return labels.iloc[0]
    if len(data.columns) == 1:  # 如果没有可分特征，返回多数类
        return labels.mode()[0]

    best_feat = best_feature(data, target)
    tree = {best_feat: {}}

    if data[best_feat].dtype == 'object':  # 离散特征
        for value in data[best_feat].unique():
            subset = data[data[best_feat] == value].drop(columns=[best_feat])
            tree[best_feat][value] = create_tree(subset, target)
    else:  # 连续特征
        # 对于连续特征，不做切分，只保留当前特征
        tree[best_feat] = 'Continuous'  # 这里我们简单标记为 "Continuous"

    return tree


# 构建决策树
decision_tree = create_tree(data, target='质量')
print("决策树:", decision_tree)



