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
X = np.hstack([X, np.ones((X.shape[0], 1))])

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


predictions = []
for i in range(len(X)):
    # 分割训练集和验证集
    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)
    X_val = X[i].reshape(1, -1)
    y_val = y[i]

    # 训练模型
    theta = np.zeros(X.shape[1])
    theta_trained = gradient_descent(X_train, y_train, theta, learning_rate=0.5, iterations=1000)

    # 预测验证集
    y_pred = predict(X_val, theta_trained)[0]
    predictions.append(y_pred)

y_pred = np.array(predictions)
accuracy = np.mean(y_pred == y) * 100
print(f"模型准确率: {accuracy:.2f}%")


plt.figure(figsize=(8, 6))
for i in range(len(X)):
    color = 'red' if y[i] == y_pred[i] else 'blue'
    marker = 'o' if y[i] == 1 else 'x'
    plt.scatter(X[i, 0], X[i, 1], color=color, marker=marker, s=100, edgecolors='k')
plt.xlabel("密度")
plt.ylabel("含糖率")
plt.title("数据分布图（红色表示预测正确，蓝色表示预测错误）")
plt.show()
