# 导入可视化工具包
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

X=load_iris().data
clf = KMeans(n_clusters=3,random_state=0)
clf.fit(X)
label = clf.predict(X)

# 颜色和标签列表
colors_list = ['red', 'blue', 'green']
labels_list = ['1','2','3']
x=X
for i in range(3):
    plt.scatter(x[label==i,0], x[label== i,1], s=100,c=colors_list[i],label=labels_list[i])
    
# 聚类中心点
plt.scatter(clf.cluster_centers_[:,0],clf.cluster_centers_[:,1], s=300,c='black',label='Centroids')

plt.legend()
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

