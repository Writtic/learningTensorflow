# -*- coding: utf-8 -*-
import tensorflow as tf
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]  # ONE-HOT encoding

X = tf.placeholder("float", [None, len(x_data[0])])
Y = tf.placeholder("float", [None, len(y_data[0])])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# logits = tf.matmul(X, W) + b
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# Tensorflow의 nn(Neural Network).softmax 로 구현
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            # optimizer 단계에서 train이 이뤄지므로 단순히 cost에 관한 sess.run은 feed에 관한 계산값만
            # 출력
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data})
    res = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9],
                                              [1, 3, 4, 3],
                                              [1, 1, 0, 1]]})
    print res, sess.run(tf.arg_max(res, 1))
    # 테스팅 & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print a, sess.run(tf.arg_max(a, 1))
