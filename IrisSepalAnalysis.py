import pandas as pd
from sklearn.cluster import KMeans  # K-means算法
import matplotlib.pyplot as plt


# 读取鸢尾花卉数据集，绘制花萼长度和花萼宽度特征之间的散点图
# 用来正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 读取csv数据集
iris_data = pd.read_csv('iris.csv')
# 构建鸢尾花卉数据
X = iris_data[['sepal_length', 'sepal_width']]
# 输出确认是否有数据缺失
print(X.shape)
# 绘制数据分布图
estimatorYiki = KMeans(n_clusters=3)  # 构造聚类器
estimatorYiki.fit(X)  # 聚类
label_pred_yiki = estimatorYiki.labels_  # 获取聚类标签
# 开始绘制K-means结果
x0 = X[label_pred_yiki == 0]

x1 = X[label_pred_yiki == 1]

x2 = X[label_pred_yiki == 2]

# 绘制鸢尾花卉散点图
plt.scatter(x0.values[:, 0], x0.values[:, 1], c='r', marker='o', label='setosa(山鸢尾)')
plt.scatter(x1.values[:, 0], x1.values[:, 1], c='g', marker='o', label='versicolor(变色鸢尾)')
plt.scatter(x2.values[:, 0], x2.values[:, 1], c='blue', marker='o', label='virgincia(维吉尼亚鸢尾)')

plt.xlabel('sepal_length(花萼长度)')
plt.ylabel('sepal_width(花萼宽度)')
plt.title('花萼长度和花萼宽度特征之间的散点图')
plt.legend(loc=2)  # 把图例放到左上角
plt.show()
