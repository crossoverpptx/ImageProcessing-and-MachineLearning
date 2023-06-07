from sklearn.model_selection import train_test_split
# 建立模型用的训练数据集和验证数据集
train_X, test_X, train_y, test_y = train_test_split(source_X , source_y, train_size=.8)

# 导入算法
from sklearn.linear_model import LogisticRegression
# 创建模型：逻辑回归（logisic regression）
model = LogisticRegression()
# 训练模型
model.fit( train_X , train_y )
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

# 评估模型
# 分类问题，score得到的是模型的正确率
model.score(test_X , test_y )
# k折交叉验证
from sklearn import model_selection
# 将训练集分成5份，4份用来训练模型，1份用来预测，这样就可以用不同的训练集在一个模型中训练
print(model_selection.cross_val_score(model, source_X, source_y, cv=5))

# 结果预测
pred_Y = model.predict(pred_X)
# 生成的预测值是浮点数（0.0,1,0）,所以要对数据类型进行转换
pred_Y=pred_Y.astype(int)
# 乘客id
passenger_id = full.loc[sourceRow:,'PassengerId']
# 数据框：乘客id，预测生存情况的值
predDf = pd.DataFrame(
    { 'PassengerId': passenger_id ,
     'Survived': pred_Y } )
predDf.shape
print(predDf.head())
# 保存结果
predDf.to_csv('titanic_pred.csv', index = False )

