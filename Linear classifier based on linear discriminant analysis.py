import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = './watermelon3_0_Ch.csv'
data = pd.read_csv(file_path, encoding='gbk')
data_numeric = data[['密度', '含糖率']]
data_numeric['好瓜'] = data['好瓜'].apply(lambda x: 1 if x == '是' else 0)
X = data_numeric[['密度', '含糖率']].values
y = data_numeric['好瓜'].values

# 分类预测函数
def predict(w, X, mean_0, mean_1):
    # 通过投影值与阈值比较进行分类
    projections = X.dot(w)
    threshold = 0.5 * (mean_0.dot(w) + mean_1.dot(w))
    return (projections > threshold).astype(int)

# 使用留一法交叉验证
predictions = []
for i in range(len(X)):
    # 划分训练集和验证集
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)
    X_val = X[i].reshape(1, -1)
    # 计算每类的均值
    mean_0 = np.mean(X_train[y_train == 0], axis=0)
    mean_1 = np.mean(X_train[y_train == 1], axis=0)

    # 初始化类内散度矩阵为零矩阵
    S_within = np.zeros((X_train.shape[1], X_train.shape[1]))
    # 计算类别0的类内散度矩阵
    for x in X_train[y_train == 0]:
        diff = (x - mean_0).reshape(-1, 1)  # 计算样本与均值的差，并转置为列向量
        S_within += diff.dot(diff.T)

    # 计算类别1的类内散度矩阵
    for x in X_train[y_train == 1]:
        diff = (x - mean_1).reshape(-1, 1)
        S_within += diff.dot(diff.T)

    # 计算投影向量w
    w = np.linalg.inv(S_within).dot(mean_1 - mean_0)

    # 进行预测并保存预测结果
    y_pred = predict(w, X_val, mean_0, mean_1)
    predictions.append(y_pred[0])

# 计算混淆矩阵和准确率
y_pred = np.array(predictions)
accuracy = np.mean(y_pred == y) * 100
print(f"模型准确率: {accuracy:.2f}%")

plt.figure(figsize=(8, 6))
for i in range(len(X)):
    color = 'red' if y[i] == y_pred[i] else 'blue'
    marker = 'o' if y[i] == 1 else 'x'
    plt.scatter(X[i, 0], X[i, 1], color=color, marker=marker, s=100)

    if i == 13:
        # 投影方向的斜率为 w[1] / w[0]
        x_k = w[1] / w[0]


        x_vals_proj = np.linspace(0, X[:, 0].max() + 0.1, 100)

        # 使用斜率绘制直线，使其从原点开始 y = slope_proj * x
        y_vals_proj = x_k * x_vals_proj
        plt.plot(x_vals_proj, y_vals_proj, 'g--', label='投影方向直线 y = w·x')

plt.xlabel("密度")
plt.ylabel("含糖率")
plt.title("数据分布图（红色表示预测正确，蓝色表示预测错误）")
plt.legend()
plt.show()




