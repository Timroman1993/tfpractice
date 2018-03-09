#coding=UTF-8
import tensorflow as tf 
import numpy as np 

#create training data : y = 0.1x + 0.3
x_data = np.random.rand(1,100) # vector 1*100
x_data.astype(np.float32)  # transforming the data to float32 for tensorflow

y_data = 0.1*x_data + 0.3

### create tensorflow structure start ###

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))   # using Weight instead of weight because Weight may be a matrix 
# shape:[1] ;  -1.0< num <1.0
biases = tf.Variable(tf.zeros([1]))


y = Weights*x_data+biases
loss = tf.reduce_mean(tf.square(y-y_data)) #compute the mean of elements across dimensions of a tensor
optimizer = tf.train.GradientDescentOptimizer(0.5) #learning rate = 0.5 
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

### create tensorflow structure  end  ###

sess = tf.Session()
sess.run(init)

for step in range(201):
	sess.run(train)
	if step % 20 == 0 :
		print 'step= %d' % step,
		print 'weight= %f' % sess.run(Weights),
		print 'bias= %f' % sess.run(biases)

