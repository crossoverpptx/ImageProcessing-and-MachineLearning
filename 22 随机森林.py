


##随机森林
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
#import numpy as np

#from sklearn.datasets import load_iris
#iris=load_iris()
##print iris#iris的４个属性是：萼片宽度　萼片长度　花瓣宽度　花瓣长度　标签是花的种类：setosa versicolour virginica
#print(iris['target'].shape)
#rf=RandomForestRegressor()#这里使用了默认的参数设置
#rf.fit(iris.data[:150],iris.target[:150])#进行模型的训练

##随机挑选两个预测不相同的样本
#instance=iris.data[[100,109]]
#print(instance)
#rf.predict(instance[[0]])
#print('instance 0 prediction；',rf.predict(instance[[0]]))
#print( 'instance 1 prediction；',rf.predict(instance[[1]]))
#print(iris.target[100],iris.target[109])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split #切分训练集和测试集
from sklearn.ensemble import RandomForestClassifier #导入随机森林模型
from sklearn import tree, datasets

#载入红酒数据
wine = datasets.load_wine()

#只选取前两个特征
X = wine.data[:, :2]
y = wine.target

#拆分训练集和数据集
X_train, X_test, y_train, y_test = train_test_split(X, y)

#设定随机森林中有6颗树
forest = RandomForestClassifier(n_estimators=6, random_state=3)

#拟合数据
forest.fit(X_train, y_train)

#绘制图形
#定义图像中分区的颜色和散点的颜色
cmap_light= ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#分别用样本的两个特征值创建图像和横轴和纵轴
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
#用不同的背景色表示不同的类
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))
z = forest.predict(np.c_[(xx.ravel(), yy.ravel())]).reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx, yy, z, cmap=cmap_light)

#用散点把样本标出来
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.title('Classifier: RandomForestClassifier') #依照参数值修改标题
plt.show()


