


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()  # 导入数据集
X = iris['data']  # 获取自变量数据
y = iris['target']  # 获取因变量数据
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2)  # 分割训练集和测试集
clf = MLPClassifier(solver='adam', alpha=1e-5, \
                    hidden_layer_sizes=(3,3), random_state=1,max_iter=100000,) # 创建神经网络分类器对象
clf.fit(X, y) # 训练模型
clf.score(X_test,y_test) # 模型评分


