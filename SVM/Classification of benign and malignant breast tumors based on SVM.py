import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd  # 用于数据加载与操作
from sklearn.model_selection import train_test_split  # 用于划分数据集
from sklearn.preprocessing import StandardScaler  # 用于特征标准化
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 加载数据
file_path = "./data.csv"
data = pd.read_csv(file_path)

# 删除无关列
data = data.drop(columns=["id", "Unnamed: 32"])

# 将目标列编码为数值，将 diagnosis 列编码为数值：M -> 1（恶性），B -> 0（良性）
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# 分离特征和标签
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"]

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 随机划分数据集，x_train训练集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 查看数据集大小
print(len(X_train), len(X_val), len(X_test))

# SVM 分类器
class SVM:
    def __init__(self, C=None, max_iter=1000, tol=1e-4, lr=0.001):
        """
        初始化 SVM 模型
        :param C: 正则化参数，None 表示硬间隔，正数表示软间隔
        :param tol: 收敛阈值
        :param lr: 学习率
        """
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr
        self.w = None  #权重
        self.b = None  #偏置

    def fit(self, X, y, lr=0.001, epochs=200):
        n_samples, n_features = X.shape
        y = np.array(y) * 2 - 1  # 将目标值转换为 {-1, 1}

        # 初始化权重和偏置
        self.w = np.zeros(n_features)
        self.b = 0
        for epoch in range(epochs):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.w) + self.b) < 1
                if self.C is None:  # 硬间隔
                    if condition:
                        self.w += lr * y[i] * X[i]
                        self.b += lr * y[i]
                else:  # 软间隔
                    if condition:
                        self.w += lr * (y[i] * X[i] - 2 * self.C * self.w)
                        self.b += lr * y[i]
                    else:
                        # 正确分类样本只更新正则项
                        self.w -= lr * 2 * self.C * self.w

    def predict(self, X):
        # 计算线性分类函数的结果
        linear_output = np.dot(X, self.w) + self.b
        # 根据符号判断类别：> 0 为 1，<= 0 为 -1
        predictions = np.sign(linear_output)
        # 将 {-1, 1} 转换为原始标签 {0, 1}
        return (predictions + 1) // 2

# 初始化 SVM 分类器
print("硬间隔 SVM 训练中...")
svm_hard = SVM(C=None, lr=0.01)
svm_hard.fit(X_train, y_train)

print("软间隔 SVM 训练中...")
best_C = None
best_accuracy = 0
best_model = None
# 定义需要测试的 C 值
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
for C in C_values:
    svm_soft = SVM(C=C, lr=0.01)
    svm_soft.fit(X_train, y_train)
    # 在验证集上评估
    y_val_pred = svm_soft.predict(X_val)
    accuracy = np.mean(y_val_pred == y_val)
    # 保存最好的模型
    if accuracy > best_accuracy:
        best_C = C
        best_accuracy = accuracy
        best_model = svm_soft
print(f"最佳软间隔参数 C={best_C}, 验证集准确率: {best_accuracy:.4f}")


# 硬间隔性能评估
y_pred_hard = svm_hard.predict(X_test)
acc_hard = accuracy_score(y_test, y_pred_hard)
recall_hard = recall_score(y_test, y_pred_hard)
f1_hard = f1_score(y_test, y_pred_hard)

# 软间隔性能评估
y_test_pred = best_model.predict(X_test)
test_accuracy = np.mean(y_test_pred == y_test)
recall_soft = recall_score(y_test, y_test_pred)
f1_soft = f1_score(y_test, y_test_pred)

print(f"硬间隔 SVM 测试集准确率: {acc_hard:.4f}, 召回率: {recall_hard:.4f}, F1 值: {f1_hard:.4f}")
print(f"软间隔SVM最佳模型 测试集准确率: {test_accuracy:.4f}, 召回率: {recall_soft:.4f}, F1 值: {f1_soft:.4f}")

# 绘制混淆矩阵
cm_hard = confusion_matrix(y_test, y_pred_hard)
cm_soft = confusion_matrix(y_test, y_test_pred)

# 显示硬间隔SVM的混淆矩阵
disp_hard_matrix = ConfusionMatrixDisplay(confusion_matrix=cm_hard, display_labels=["malignant (1)", "benign (0)"])
disp_hard_matrix.plot(cmap=plt.cm.Blues)
plt.title("HARD_SVM confusion_matrix")
plt.show()

# 显示软间隔SVM的混淆矩阵
disp_soft_matrix = ConfusionMatrixDisplay(confusion_matrix=cm_soft, display_labels=["malignant (1)", "benign (0)"])
disp_soft_matrix.plot(cmap=plt.cm.Blues)
plt.title("SOFT_SVM confusion_matrix")
plt.show()














