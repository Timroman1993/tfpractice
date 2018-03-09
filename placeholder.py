import tensorflow as tf 

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)

do_multiply = tf.mul(x1,x2)

with tf.Session() as sess:
	print sess.run(do_multiply,feed_dict={x1:7.,x2:2.})
	print sess.run(do_multiply,feed_dict={x1:[7.],x2:[2.]})