import tensorflow as tf


'''
state = tf.Variable(0, name='counter')
#print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)


init = tf.initialize_all_variables()  # must have if define variable
'''
'''
with tf.Session() as sess:
	sess.run(init)
	print sess.run(new_value)
'''
'''
update = tf.assign(state, new_value)

with tf.Session() as sess:
	sess.run(init)
	sess.run(update)
	print sess.run(new_value)
	print sess.run(state)

'''

counter = tf.Variable(0,name='counter')
one = tf.constant(1)
times = tf.add(counter,one) 
update = tf.assign(counter,times)  # counter++
init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for i in range(3):
		sess.run(update)
		print sess.run(counter)
