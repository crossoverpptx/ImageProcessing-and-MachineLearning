


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
 
iris = load_iris()#数据集导入
features = iris.data#属性特征
labels = iris.target#分类标签
train_features, test_features, train_labels, test_labels = \
    train_test_split(features, labels, test_size=0.3, random_state=1)#训练集，测试集分类
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf = clf.fit(train_features, train_labels)#X,Y分别是属性特征和分类label
test_labels_predict = clf.predict(test_features)# 预测测试集的标签
score = accuracy_score(test_labels, test_labels_predict)# 将预测后的结果与实际结果进行对比
print("CART分类树的准确率 %.4lf" % score)# 输出结果
dot_data = tree.export_graphviz(clf, out_file='iris_tree.dot')#生成决策树可视化的dot文件


