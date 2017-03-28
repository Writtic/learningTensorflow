# -*- coding: utf-8 -*-
import tensorflow as tf

# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# shape=[None] : 아무 shape이나 들어올 수 있다
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# tf.Variable : 텐서플로우가 트레이닝하면서 변경할 수 있는 변수(Trainable Variable)
# tf.random_normal : rank와 shape이 1(값이 하나인 1차원 어레이)인 랜덤 값을 리턴
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = x_train * W + b
hypothesis = X * W + b
# tf.square : 제곱을 리턴
# tf.reduce_mean : 평균값 리턴
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
# tf.Variable을 사용하기 위해선 꼭 tf.global_variables_initializer() 로 초기화
sess.run(tf.global_variables_initializer())

temp_x = [, 2, 3, 4, 5]
temp_y = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for step in xrange(2001):
    # sess.run(train)
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: temp_x, Y: temp_y})
    if step % 20 == 0:
        print step, cost_val, W_val, b_val
        # print step, sess.run(cost), sess.run(W), sess.run(b)

# Test the model we created
print sess.run(hypothesis, feed_dict={X: [5]})
print sess.run(hypothesis, feed_dict={X: [2, 3]})
# The output is gonna be Y value
