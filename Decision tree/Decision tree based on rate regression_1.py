import pandas as pd
import numpy as np

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


# 自定义的对率回归模型
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    return sigmoid(X.dot(theta)) >= 0.5

def compute_loss(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    loss = (-1 / m) * (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h)))
    return loss

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta = theta - learning_rate * gradient
    return theta


# 用自定义的对率回归模型进行划分选择
def logistic_regression_split(data, feature, target):
    if data[feature].dtype in ['float64', 'int64']:
        thresholds = sorted(data[feature].unique())
    else:
        return None, 0

    best_accuracy = 0
    best_threshold = None

    for threshold in thresholds:
        left = data[data[feature] <= threshold]
        right = data[data[feature] > threshold]

        if len(left) == 0 or len(right) == 0:
            continue

        # 对数回归训练
        X_train = left[feature].values.reshape(-1, 1)
        y_train = left[target]

        theta = np.zeros(X_train.shape[1])  # 初始化参数
        theta_trained = gradient_descent(X_train, y_train, theta, learning_rate=0.5, iterations=1000)

        # 验证模型
        X_val = right[feature].values.reshape(-1, 1)
        y_val = right[target]
        y_pred = predict(X_val, theta_trained)

        accuracy = np.mean(y_pred == y_val)  # 计算准确率

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


# 选择最佳特征及划分点
def best_feature_by_logistic_regression(data, target, max_depth, current_depth, min_samples_split):
    features = data.columns.drop(target)
    best_feature = None
    best_threshold = None
    best_accuracy = 0

    if current_depth >= max_depth or len(data) < min_samples_split:
        return None, None, 0

    for feature in features:
        threshold, accuracy = logistic_regression_split(data, feature, target)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold, best_accuracy


# 创建决策树
def create_tree(data, target, max_depth=3, min_samples_split=2, current_depth=0):
    labels = data[target]
    if len(labels.unique()) == 1:  # 如果所有标签相同，返回叶节点
        return labels.iloc[0]
    if len(data.columns) == 1:  # 如果没有可分特征，返回多数类
        return labels.mode()[0]

    best_feature, best_threshold, best_accuracy = best_feature_by_logistic_regression(
        data, target, max_depth, current_depth, min_samples_split
    )
    if best_accuracy == 0:
        return labels.mode()[0]  # 如果没有能提供足够准确的划分，返回多数类

    tree = {best_feature: {}}

    left = data[data[best_feature] <= best_threshold]
    right = data[data[best_feature] > best_threshold]

    tree[best_feature][f"≤{best_threshold}"] = create_tree(left, target, max_depth, min_samples_split,
                                                           current_depth + 1)
    tree[best_feature][f">{best_threshold}"] = create_tree(right, target, max_depth, min_samples_split,
                                                           current_depth + 1)
    return tree


# 构建决策树
decision_tree = create_tree(data, target='质量', max_depth=4, min_samples_split=2)
print(decision_tree)


