from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

'''

sklearn工具箱实现

'''
#参数：
# n_samples=100  样本数量
# n_features=2   特征数量
# centers=3      中心点

#返回值：
# X_train:  测试集
# y_train： 特征值

X_train,y_train = make_blobs(n_samples=100, n_features=2, centers=3)

# 再生成一组作为测试集使用，设置43个中心，主打一个杂乱无章
X, y = make_blobs(n_samples=200, centers=43, cluster_std=0.60, random_state=0)

# 参数
# n_clusters  将预测结果分为几簇

kmeans = KMeans(n_clusters=3)  # 获取模型
kmeans.fit(X_train)  # 要分类的数据给他，对模型进行训练

y_ = kmeans.predict(X)
print(y_) # 预测结果
# 画出还未进行划分的预测集
plt.scatter(X[:,0],X[:,1])
plt.figure()
# 训练集的点加答案
plt.scatter(X_train[:,0],X_train[:,1],c=y_train) # 原结果
plt.figure()
# 预测结果的的点和对应所属的聚合
plt.scatter(X[:,0],X[:,1],c=y_)  # 预测结果

plt.show()