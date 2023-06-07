

# 导入包
from sklearn.naive_bayes import GaussianNB  # 高斯分布，假定特征服从正态分布的
from sklearn.model_selection import train_test_split # 数据集划分
from sklearn.metrics import accuracy_score

# 导入数据集
from sklearn import datasets
iris = datasets.load_iris()

# 拆分数据集,random_state:随机数种子
train_x,test_x,train_y,test_y = train_test_split(iris.data,iris.target,random_state=12) 

# 建模
gnb_clf = GaussianNB()
gnb_clf.fit(train_x,train_y)

# 对测试集进行预测
# predict()：直接给出预测的类别
# predict_proba()：输出的是每个样本属于某种类别的概率
predict_class = gnb_clf.predict(test_x)
# predict_class_proba = gnb_clf.predict_proba(test_x)
print("测试集准确率为：",accuracy_score(test_y,predict_class))

