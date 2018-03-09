#coding=utf-8
import tensorflow as tf 

def add_layer(inputs,in_size,out_size,activation_function=None): #用激励函数处理

	Weights = tf.Variable(tf.random_normal([in_size,out_size]),name = 'Weights') #权重为一个矩阵，shape为 in_size * out_size
	biases = tf.Variable(tf.zeros([1,out_size]) +0.1 ) #初始化为0 不好 ，+0.1调整一下
	Wx_plus_b = tf.matmul(Weights,inputs) + biases
	if activation_function is None :
		outputs = Wx_plus_b
	else :
		outputs = activation_function(Wx_plus_b)

	return outputs
