import cifar10,cifar10_input
import tensorflow as tf
import numpy as np
import time

max_steps = 3000#定义训练轮数
batch_size = 128#
data_dir = 'D:\\tmp\\cifar10_data\cifar-10-batches-bin'

def variable_with_weight_loss(shape,stddev,w1):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weight_loss')#l2_loss*w1
        tf.add_to_collection('losses',weight_loss)#把weight_loss添加到一个collection，并命名为'losses'
    return var

cifar10.maybe_download_and_extract()#使用cifar10类下载数据集，并解压、展开到默认位置
images_train,labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)#产生训练数据
images_test,labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)#产生测试数据
#创建输入的placeholder，包括特征和label
image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3])
label_holder = tf.placeholder(tf.int32,[batch_size])
#设置第一个卷积层
weight1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)#卷积核3*3,3颜色通道，64个卷积核,不进行L2正则化
kernel1 = tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME')#对iamge_holder进行卷积操作，步长均为1
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))#偏置设置为0
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))#Relu激活函数
pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')#最大池化函数
norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)#LRN对结果进行处理
#创建第二个卷积层
weight2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
kernel2 = tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME')#卷积
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))#bias值为0.1
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)#先lrn处理再最大池化
pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
#设置一个全连接层，
reshape = tf.reshape(pool2,[batch_size,-1])#需要先将前面两个卷积层的输出结果全部flatten，使用tf.reshape将向量一维化
dim = reshape.get_shape()[1].value#利用get_shape()获取数据扁平化之后的长度
weight3 = variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
bias3 = tf.Variable(tf.constant(0.1,shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape,weight3)+bias3)
#全连接层
weight4 = variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
bias4 = tf.Variable(tf.constant(0.1,shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3,weight4)+bias4)
#输出层
weight5 = variable_with_weight_loss(shape=[192,10],stddev=1/192.0,w1=0.0)
bias5 = tf.Variable(tf.constant(0.0,shape=[10]))
logits = tf.add(tf.matmul(local4,weight5),bias5)
#loss of CNN
def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(#将softmax和cross entropy loss计算在一起
        logits=logits,labels=labels,name='cross_entrop_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')#求均值
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')#将整体losses的collection中的全部loss求和，最终得到的loss

loss = loss(logits,label_holder)#将logits节点和label_placeholder传入函数获得最终的loss
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)#选择Adam Optimizer优化器，学习速率为1e-3
top_k_op = tf.nn.in_top_k(logits,label_holder,1)#求出结果中top k的准确率，默认使用top 1，也就是输出分数最高的那一类
#创建session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()#初始化全部模型参数
tf.train.start_queue_runners()
#开始训练
for step in range(max_steps):
    start_time = time.time()
    image_batch,label_batach = sess.run([images_train,labels_train])
    _,loss_value = sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batach})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)')
        print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))
num_examples = 100
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch,label_batach = sess.run([images_test,labels_test])
    predictions = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batach})
    true_count += np.sum(predictions)
    step += 1
precision = true_count / total_sample_count
print('precision @ 1 = %.3f'%precision)


