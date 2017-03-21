import tensorflow as tf
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(5.)

# Custom gradient descent optimizer
hypothesis = X * W
gradient = tf.reduce_mean((W * X - Y) * X) * 2  # derivative
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Tensorflow's gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(100):
    our_g, our_w, tensor_val = sess.run([gradient, W, gvs])
    print step, "Custom: ", our_g, our_w, "Tensor: ", tensor_val
    sess.run(apply_gradients)
