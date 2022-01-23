import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


w = tf.Variable(tf.random_normal([1]), name = 'weight')    # Variable used by TF
b = tf.Variable(tf.random_normal([1]), name = 'bias')      

hypo = x_train * w + b

cost = tf.reduce_mean(tf.square(hypo - y_train))



opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = opt.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
  cost_val, w_val, b_val , _ = sess.run([cost, w, b, train], feed_dict = {X: [1, 2, 3], Y : [2, 4, 6]})
  
