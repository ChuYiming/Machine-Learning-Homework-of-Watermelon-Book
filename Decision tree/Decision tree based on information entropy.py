import pandas as pd
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


# 计算信息熵
def entropy(labels):
    label_counts = labels.value_counts()
    total = len(labels)
    return sum(-count / total * math.log2(count / total) for count in label_counts if count > 0)


# 计算离散特征的信息增益
def info_gain_discrete(data, feature, target):
    original_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = sum(
        (len(subset := data[data[feature] == value]) / len(data)) * entropy(subset[target])
        for value in values
    )
    return original_entropy - weighted_entropy


# 计算连续特征的信息增益
def info_gain_continuous(data, feature, target):
    sorted_data = data.sort_values(by=feature)
    thresholds = sorted_data[feature].rolling(2).mean().dropna()  # 计算候选分割点
    original_entropy = entropy(data[target])
    max_gain = -1
    best_threshold = None
    for threshold in thresholds:
        left = data[data[feature] <= threshold]
        right = data[data[feature] > threshold]
        weighted_entropy = (
                len(left) / len(data) * entropy(left[target]) +
                len(right) / len(data) * entropy(right[target])
        )
        gain = original_entropy - weighted_entropy
        if gain > max_gain:
            max_gain = gain
            best_threshold = threshold
    return max_gain, best_threshold


# 选择最佳特征
def best_feature(data, target):
    features = data.columns.drop(target)
    max_gain = -1
    best_feat = None
    best_split = None
    for feature in features:
        if data[feature].dtype == 'object':  # 离散值
            gain = info_gain_discrete(data, feature, target)
            split = None
        else:  # 连续值
            gain, split = info_gain_continuous(data, feature, target)
        if gain > max_gain:
            max_gain = gain
            best_feat = feature
            best_split = split
    return best_feat, best_split


# 构建决策树
def create_tree(data, target):
    labels = data[target]
    if len(labels.unique()) == 1:  # 如果所有标签相同，返回叶节点
        return labels.iloc[0]
    if len(data.columns) == 1:  # 如果没有可分特征，返回多数类
        return labels.mode()[0]
    best_feat, best_split = best_feature(data, target)
    tree = {best_feat: {}}
    if best_split is None:  # 离散值
        for value in data[best_feat].unique():
            subset = data[data[best_feat] == value].drop(columns=[best_feat])
            tree[best_feat][value] = create_tree(subset, target)
    else:  # 连续值
        left = data[data[best_feat] <= best_split]
        right = data[data[best_feat] > best_split]
        tree[best_feat][f"≤{best_split:.3f}"] = create_tree(left, target)
        tree[best_feat][f">{best_split:.3f}"] = create_tree(right, target)
    return tree


# 构建决策树
decision_tree = create_tree(data, target='质量')
print("决策树:", decision_tree)







