from sklearn import metrics
import numpy as np
import networkx as nx
from sklearn.preprocessing import OrdinalEncoder

#生成树算法，TAN使用最大权生成树
class MST:
    def __init__(self, tree_type, algorithm):
        # tree_type: 'max' 表示最大生成树，'min' 表示最小生成树
        # algorithm: 'Kruskal' 或 'Prim'，表示使用的算法
        self.tree_type = tree_type
        self.algorithm = algorithm

    def fit_transform(self, points, CMI_dict):
        # 创建图
        G = nx.Graph()
        # 添加节点
        G.add_nodes_from(points)
        # 根据CMI_dict条件互信息添加边
        for (i, j), weight in CMI_dict.items():
            if self.tree_type == 'max':
                # 如果是最大生成树，则边的权重取负数，Kruskal会选择较小的负权重
                G.add_edge(i, j, weight=-weight)
            else:
                # 最小生成树的情况
                G.add_edge(i, j, weight=weight)
        # 使用 Kruskal 算法计算最大生成树
        if self.algorithm == 'Kruskal':
            mst_edges = list(nx.minimum_spanning_edges(G, algorithm='kruskal', data=True))
        else:
            raise ValueError("Unsupported algorithm. Only 'Kruskal' is supported for now")
        # 返回生成树的边列表（每个元素是 (node1, node2, weight)）
        mst = [(u, v, -w['weight']) for u, v, w in mst_edges]
        return mst

# TAN算法,树增强型贝叶斯（Tree Augmented Naive Bayes）
class Tan(object):
    def __init__(self):
        self.trainSet = 0  # 训练集数据
        self.trainLabel = 0  # 训练集标记
        self.yProba = {}  # 先验概率容器
        self.xyProba = {}  # 条件概率容器
        self.ySet = {}  # 标记类别对应的数量
        self.ls = 1  # 加入的拉普拉斯平滑的系数
        self.n_samples = 0  # 训练集样本数量
        self.n_features = 0  # 训练集特征数量
        self.CMI_dict = dict()  # 条件互信息字典
        self.f_relationship = dict()  # 特征依赖关系{子特征：父特征（依赖属性）}
    # 计算P(y)先验概率
    def calPy(self, y, LS=True):
        Py = {}
        yi = {}
        ySet = np.unique(y)
        for i in ySet:
            Py[i] = (sum(y == i) + self.ls) / (self.n_samples + len(ySet))
            yi[i] = sum(y == i)
        self.yProba = Py
        self.ySet = yi
        return

    # 离散变量直接计算概率
    def classifyProba(self, x, xArr, XiSetCount):
        Pxy = (sum(xArr == x) + self.ls) / (xArr.size + XiSetCount)  # 加入拉普拉斯修正的概率
        return Pxy

    # 基于父属性计算离散特征的条件概率
    def __categorytrain(self, xarr, xiset):
        pxypa = {}
        for xivalue in xiset:
            pxypa[xivalue] = {}
            pxypa[xivalue]['count'] = sum(xarr == xivalue) + self.ls
            pxypa[xivalue]['ratio'] = self.classifyProba(xivalue, xarr, len(xiset))
        return pxypa

    # 计算连续特征的均值和标准差
    def __continuoustrain(self, Xarr):
        pxypa = (Xarr.mean(), Xarr.std())
        return pxypa

    # 计算两个离散特征的条件互信息
    def __CMI_classfic(self, xi, xj, y):
        yset = np.unique(y)
        y_count = y.size
        cmi = 0
        for yi in yset:
            yi_idx = np.nonzero(y == yi)[0]
            yi_count = yi_idx.size
            yi_proba = yi_count / y_count
            arr0 = xi[yi_idx]
            arr1 = xj[yi_idx]
            # 计算两个离散特征的互信息
            mi = metrics.mutual_info_score(arr0, arr1)
            cmi += mi * yi_proba
        return cmi

    # 计算特征之间的条件互信息，生成离散特征之间的依赖关系，并生成字典
    def get_cmidict(self, X, y, columnsMark):
        n_samples, n_features = X.shape
        CMI_dict = dict()
        featureIdx = np.nonzero(np.array(columnsMark) == 0)[0]
        for idx, i in enumerate(featureIdx[:-1]):
            for j in featureIdx[idx + 1:]:
                # print(i, j)
                CMI_dict[(i, j)] = self.__CMI_classfic(X[:, i], X[:, j], y)
        return CMI_dict

    # 获取特征的依赖关系
    def get_relationship(self, features: list):
        # 获取最大次数的顶点
        point_count = dict()
        for p0, p1 in features:
            if p0 not in point_count.keys():
                point_count[p0] = 1
            else:
                point_count[p0] += 1
            if p1 not in point_count.keys():
                point_count[p1] = 1
            else:
                point_count[p1] += 1
        pointcntList = sorted(point_count.items(), key=lambda x: x[1], reverse=True)
        maxcntFeature = pointcntList[0][0]
        # 遍历特征，保存依赖关系
        feature_relationship = dict()
        feature_epoch = [maxcntFeature]
        features_index = []
        while len(features_index) < len(features):
            for idx, feature_pair in enumerate(features):
                if idx in features_index:
                    continue
                if feature_pair[0] in feature_epoch:
                    feature_relationship[feature_pair[1]] = feature_pair[0]
                    feature_epoch.append(feature_pair[1])
                    features_index.append(idx)
                elif feature_pair[1] in feature_epoch:
                    feature_relationship[feature_pair[0]] = feature_pair[1]
                    feature_epoch.append(feature_pair[0])
                    features_index.append(idx)
                else:
                    continue
        return feature_relationship

    # TAN算法训练
    def TanTrain(self, X, y, columnsMark):
        # 1 根据最大带权生成树找到每个特征的依赖属性，获得离散特征父属性
        ## 1.1 计算互信息字典
        self.CMI_dict = self.get_cmidict(X, y, columnsMark)
        ## 1.2 生成最大带权树
        points = list(set([i[0] for i in self.CMI_dict.keys()] + [i[1] for i in self.CMI_dict.keys()]))
        self.MaxstClass = MST('max', 'Kruskal')
        self.Maxspanningtree = self.MaxstClass.fit_transform(points, self.CMI_dict)
        ## 1.3自定义顶点，使之有向，构建出依赖关系
        self.f_relationship = self.get_relationship([(f1, f2) for f1, f2, w in self.Maxspanningtree])

        # 2 训练贝叶斯的联合概率、条件概率
        ## 2.1 初始化变量，样本数量、特征数量、标签类别、
        ##     类别的先验联合概率、联合概率的的条件概率
        self.n_samples, self.n_features = X.shape
        # 计算类别的先验概率
        self.calPy(y)
        print('P(y)训练完毕!')
        Pxypa = {}
        # 第一层是不同的分类
        for yi, yiCount in self.ySet.items():
            Pxypa[yi] = {}
            # 第二层是不同的特征，如果有父属性，就接着加一层父属性，如果没有父属性，则按朴素贝叶斯公式
            for xiIdx in range(self.n_features):
                allXiset = np.unique(X[:, xiIdx])
                # 没有父属性的
                if xiIdx not in self.f_relationship.keys():
                    Xiarr = X[np.nonzero(y == yi)[0], xiIdx].flatten()
                    if columnsMark[xiIdx] == 0:
                        ## 保存离散特征的条件概率
                        Pxypa[yi][xiIdx] = self.__categorytrain(Xiarr, allXiset)
                    else:
                        ## 保存连续特征的条件概率
                        Pxypa[yi][xiIdx] = self.__continuoustrain(Xiarr)
                    continue

                # 第三层是有父属性的，值为父属性的各类值
                Pxypa[yi][xiIdx] = {}
                paIdx = self.f_relationship[xiIdx]
                paset = np.unique(X[:, paIdx])
                for pai in paset:
                    # 在计算存在依赖属性的条件概率时，获得父属性为特定值的子属性，
                    # 再输入到朴素贝叶斯条件概率计算公式，即实现TAN条件概率算法
                    xi_pai_idx = np.nonzero((X[:, paIdx] == pai) & (y == yi))[0]
                    Xiarr = X[xi_pai_idx, xiIdx].flatten()
                    if columnsMark[xiIdx] == 0:
                        ## 保存离散特征的条件概率
                        Pxypa[yi][xiIdx][pai] = self.__categorytrain(Xiarr, allXiset)
                    else:
                        ## 保存连续特征的条件概率
                        Pxypa[yi][xiIdx][pai] = self.__continuoustrain(Xiarr)
        print('P(x|y,pa)训练完毕!')
        self.xyProba = Pxypa
        self.trainSet = X
        self.trainLabel = y
        self.columnsMark = columnsMark
        return

    # 预测
    def tanpredict(self, X):
        n_samples, n_features = X.shape
        proba = np.zeros((n_samples, len(self.yProba)))
        for i in range(n_samples):
            for idx, (yi, Xidict) in enumerate(self.xyProba.items()):
                probaValue = 1.
                probaValue *= self.yProba[yi]
                for xiIdx, valuedict in Xidict.items():
                    xi = X[i, xiIdx]
                    ## 值不是字典，说明是连续变量
                    if not isinstance(valuedict, dict):
                        miu = valuedict[0];
                        sigma = valuedict[1] + 1.0e-5
                        Pxypa = np.exp(-(xi - miu) ** 2 / (2 * sigma ** 2)) / (
                                    np.power(2 * np.pi, 0.5) * sigma) + 1.0e-5
                    ## 第三层不是字典，说明特征没有依赖属性
                    elif not isinstance(list(list(valuedict.values())[0].values())[0], dict):
                        Pxypa = valuedict[xi]['ratio']
                    ## 第三层是字典，说明有依赖属性
                    else:
                        pai = X[i, self.f_relationship[xiIdx]]
                        Pxypa = valuedict[pai][xi]['ratio']
                    probaValue *= Pxypa
                proba[i, idx] = probaValue
        return proba

    # 取对数预测
    def tanpredictLog(self, X):
        n_samples, n_features = X.shape
        proba_log = np.zeros((n_samples, len(self.yProba)))
        for i in range(n_samples):
            for idx, (yi, Xidict) in enumerate(self.xyProba.items()):
                probaValueLog = 0.
                probaValueLog += np.log(self.yProba[yi])
                for xiIdx, valuedict in Xidict.items():
                    xi = X[i, xiIdx]
                    ## 值不是字典，说明是连续变量
                    if not isinstance(valuedict, dict):
                        miu = valuedict[0]
                        sigma = valuedict[1] + 1.0e-5
                        Pxypa = np.exp(-(xi - miu) ** 2 / (2 * sigma ** 2)) / (
                                    np.power(2 * np.pi, 0.5) * sigma) + 1.0e-5
                    ## 第三层不是字典，说明特征没有依赖属性
                    elif not isinstance(list(list(valuedict.values())[0].values())[0], dict):
                        Pxypa = valuedict[xi]['ratio']
                    ## 第三层是字典，说明有依赖属性
                    else:
                        pai = X[i, self.f_relationship[xiIdx]]
                        Pxypa = valuedict[pai][xi]['ratio']
                    probaValueLog += np.log(Pxypa)
                proba_log[i, idx] = probaValueLog
        return proba_log

dataSet_train = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']
]
# 特征值列表
labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '密度', '含糖率']

dataSet_test = [
    ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460],
    ['浅白', '蜷缩', '沉闷', '模糊', '稍凹', '软粘', 0.463, 0.135],
    ['乌黑', '稍蜷', '清脆', '模糊', '凹陷', '硬滑', 0.428, 0.208]
]

dataX_train = np.array(dataSet_train)[:, :6]
oriencode_train = OrdinalEncoder(categories='auto')
oriencode_train.fit(dataX_train)
X_train = oriencode_train.transform(dataX_train)
X2_train = np.array(dataSet_train)[:, 6:8].astype(float)
X_train = np.hstack((X_train, X2_train))
y_train = np.array(dataSet_train)[:, 8]
y_train[y_train == "好瓜"] = 1
y_train[y_train == "坏瓜"] = 0
y_train = y_train.astype(float)

dataX_test = np.array(dataSet_test)[:, :6]
oriencode_test = OrdinalEncoder(categories='auto')
oriencode_test.fit(dataX_test)
X_test = oriencode_test.transform(dataX_test)
X2_test = np.array(dataSet_test)[:, 6:8].astype(float)
X_test = np.hstack((X_test, X2_test))

#训练
tan = Tan()
tan.TanTrain(X_train, y_train, [0, 0, 0, 0, 0, 0, 1, 1])
Proba_train = tan.tanpredict(X_train)
logProba_train = tan.tanpredictLog(X_train)
yPredict_train = np.argmax(logProba_train, axis=1)
print(f"在训练集上错误{sum(yPredict_train != y_train)}个，训练结果准确率为：{sum(yPredict_train == y_train) / y_train.size}")
#测试
Proba_test = tan.tanpredict(X_test)
logProba_test = tan.tanpredictLog(X_test)
yPredict_test = np.argmax(logProba_test, axis=1)
print("\n预测结果：")
for i in range(len(X_test)):
    print(f"样本 {i + 1}: 预测后验概率 = {logProba_test[i]}  预测标签 = {yPredict_test[i]}")






