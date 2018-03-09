import tensorflow as tf 
matrix1 = tf.constant( [ [3,3] ] )
matrix2 = tf.constant( [ [2],
                         [2] ] )
product = tf.matmul(matrix1,matrix2)
''' 
 # method1
sess = tf.Session()
print sess.run(matrix1)
print sess.run(matrix2)
print 'result=',
print sess.run(product)
sess.close()
'''

# method2
with tf.Session() as sess:
	result = sess.run(product)
	print result
