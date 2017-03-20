# -*- coding: utf-8 -*-
import tensorflow as tf
x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 선형 모델 X * W 을 위한 우리의 Hypothesis
hypothesis = X * W

cost = tf.reduce_sum(tf.square(hypothesis - Y))

# W 최소값 구하기: 기울기를 이용한 경사하강법: W -= Learning_rate * derivative
# 이전 파일의 optimizer와 동일한 기능
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
# 텐서플로우는 바로 =(이퀄)을 통해 변수값을 변경할 수 없고 assign 함수를 통해 변경해야한다.
update = W.assign(descent)

# 세션에서 설계한 그래프를 실행
sess = tf.Session()
# 그래프의 전역 변수들을 초기화
sess.run(tf.global_variables_initializer())

for step in xrange(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)
