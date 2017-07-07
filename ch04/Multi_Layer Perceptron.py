from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
sess = tf.InteractiveSession()

in_units = 784
h1_units = 300
w1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units ]))
w2 = tf.Variable(tf.zeros([h1_units,10]))

