# filename_queue = tf.train.string_input_producer(['file01.csv', 'file02.csv', ...], shuffle=False, name='filename_queue')
# record_defaults = [[0.], [0.], [0.], [0.]]
# xy = tf.decode_csv(value, record_defaults=record_defaults)
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
#
# # collect batches of csv in
# # batch_size : 한번에 가져올 사이즈
# train_x_batch, train_y_batch = \
# tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
#
# sess = tf.Session()
# ...
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

filename_queue = tf.train.string_input_producer(
    ['data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# shuffle_batch usage
# min_after_dequeue = 10000 # 버퍼 크기를 어떻게 할 것인지 정함 크면 클수록 시작 속도가 느리고, 메모리 사용이 커짐
# capacity = min_after_dequeue + 3 * batch_size # 용량은 반드시 min_after_dequeue보다 커야함 주로 min_after_dequeue + (num_thread + 에러방지용 작은 공간) * batch_size
# example_batch, label_batch = tf.train.shuffle_batch(
# [example, label], batch_size=batch_size, capacity=capacity,
# min_after_dequeue=min_after_dequeue)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.
# 일반적으로 reader와 짝을 이루는 코드
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# 일반적으로 reader와 짝을 이루는 코드
coord.request_stop()
coord.join(threads)

# Ask my score
print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

'''
Your score will be  [[ 177.78144836]]
Other scores will be  [[ 141.10997009]
 [ 191.17378235]]

'''
