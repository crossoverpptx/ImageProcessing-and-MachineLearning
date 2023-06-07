#import numpy as np

#x=np.random.random(10)
#y=np.random.random(10)

## 方法一：根据公式求解
#d1=np.sqrt(np.sum(np.square(x-y)))
 
## 方法二：根据scipy库求解
#from scipy.spatial.distance import pdist
#X=np.vstack([x,y])  # 将x，y两个一维数组合并成一个2D数组：[[x1,x2,x3...],[y1,y2,y3...]]
#d2=pdist(X) # d2=np.sqrt(（x1-y1)+(x2-y2)+....)

## 方法三：使用numpy.linalg.norm(x, y)可以计算向量 x 和向量 y 的欧氏距离
#d3=np.linalg.norm(x-y)

#print(d1, d2, d3)

import numpy as np

x=np.random.random(10)
y=np.random.random(10)

# 方法一：scipy中的scipy.spatial.distance.cosine函数可计算余弦距离。
# 因此，我们可以用1减去余弦距离得到余弦相似度。
from scipy import spatial
res1 = 1 - spatial.distance.cosine(x, y)
print(res1)

# 方法二：numpy中的numpy.dot函数可以两个向量的点积，numpy.linalg.norm函数可以计算向量的欧氏距离。
# 因此，可以通过公式和这两个函数计算向量的余弦相似度。
from numpy import dot
from numpy.linalg import norm
res2 = dot(x, y) / (norm(x) * norm(y))
print(res2)

# 方法三：sklearn中的sklearn.metrics.pairwise.cosine_similarity函数可直接计算出两个向量的余弦相似度。
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
res3 = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1)) # reshape(1, -1)将矩阵转化成1行
print(res3)



