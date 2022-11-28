


#卷积函数
import tensorflow as tf
import numpy as np

# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
input_data = tf.Variable(np.random.rand(10, 9, 9, 4), dtype=np.float32)
filter_data =  tf.Variable(np.random.rand(3, 3, 4, 2), dtype=np.float32)
# y = tf.nn.conv2d(input_data, filter_data, strides=[1,1,1,1], padding = 'SAME')
y = tf.nn.conv2d(input_data, filter_data, strides=[1,1,1,1], padding = 'VALID')

print(input_data)
print(y)

#池化函数
import tensorflow as tf
import numpy as np

input_data = tf.Variable(np.random.rand(10, 6, 6, 4), dtype=np.float32)
filter_data =  tf.Variable(np.random.rand(2, 2, 4, 2), dtype=np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides=[1,1,1,1], padding = 'SAME')

# 最大池化
# tf.nn.max_pool(value, ksize, strides, padding, name=None) 
output = tf.nn.max_pool(value=y,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

# 平均池化
# tf.nn.avg_pool(value, ksize, strides, padding, name=None)
# output = tf.nn.avg_pool(value=y,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

print('conv:',y)
print('pool_padding_valid:',output)