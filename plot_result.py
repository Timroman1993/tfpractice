#coding=utf-8
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs




### bulid training data ###
x_data = np.linspace(-1,1,300)[:,np.newaxis]
#print x_data.shape
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise  # y = x*x-0.5 上下波动
### bulid training data ###

#plot real data

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion() #自动往下执行
plt.show()


#定义出一会要喂到网络中的数据坑
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])


#TensorFlow中，输入数据有多少，就认为input多少神经元，在这里，输入层的神经元个数为1 --> 就x_data一个数据； 输出层的神经元个数为1 --> 就y_data一个数据
#设置隐藏层有10个神经元

hidden = add_layer(xs,1,10,activation_function=tf.nn.relu)
predict = add_layer(hidden,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(predict-ys),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

'''
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())

	for i in range(1000):
		sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
        if i % 50 == 0 :
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            predict_value = sess.run(predict,feed_dict={xs:x_data})
            lines = ax.plot(x_data, predict_value, 'r-', lw=5)
            plt.pause(0.5)


'''
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())    
    for i in range(1000):

    # training
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
        # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(predict, feed_dict={xs: x_data})
        # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.5)
      













