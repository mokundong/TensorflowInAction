import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()
#定义权重
def weught_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#分布类型，标准差0.1
    return tf.Variable(initial)
#定义偏置
def bias_variable(shape):
    initial = tf.concat(0.1,shape)#添加正值0.1避免死亡节点
    return tf.Variable(initial)
#创建2维卷积函数
def  conv2d(x,W):#x:输入，W是卷积参数
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#serides代表卷积模板移动的步长，都是1代表不遗漏的划过每一个点