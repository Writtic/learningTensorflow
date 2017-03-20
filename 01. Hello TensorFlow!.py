import tensorflow as tf

hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()

print(sess.run(hello))
# constant
node_1 = tf.constant(3.0, tf.float32)
node_2 = tf.constant(4.0)
node_3 = tf.add(node_1, node_2)

print("node1: ", node_1, " node2: ", node_2)
print("node3: ", node_3)
print("sess.run(node1, node2): ", sess.run([node_1, node_2]))
print("sess.run(node3): ", sess.run(node_3))

# placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

print sess.run(adder_node, feed_dict={a: 3, b: 4.5})
print sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]})


개념 : 데이터 플로우 그래프라는 것을 사용해서 수적인 계산을 한다.

데이터 플로우 : 노드(오퍼레이션), 엣지(데이터, 텐서)로 이뤄진 플로우 차트


{"kernelspecs":{"python2":{"spec":{"language":"python","argv":["/usr/local/opt/python/bin/python2.7","-m","ipykernel","-f","{connection_file}"],"display_name":"Python 2","env":{}},"resource_dir":"/Users/wriitc/Library/Jupyter/kernels/python2"},"pyspark":{"spec":{"language":"python","argv":["/Users/wriitc/.pyenv/versions/3.5.1/bin/python3.5","-m","IPython.kernel","--profile=pyspark","-f","{connection_file}"],"display_name":"pySpark (Spark 1.6.1)","env":{"PYSPARK_SUBMIT_ARGS":"--conf 'spark.mesos.coarse=true' pyspark-shell","SPARK_HOME":"/Users/wriitc/spark-1.6.1-bin-hadoop2.6"}},"resource_dir":"/Users/wriitc/.ipython/kernels/pyspark"},"python3":{"spec":{"language":"python","argv":["/Users/wriitc/.pyenv/versions/3.5.1/bin/python3.5","-m","ipykernel","-f","{connection_file}"],"display_name":"Python 3","env":{}},"resource_dir":"/Users/wriitc/.ipython/kernels/python3"},"roguelike":{"spec":{"language":"python","argv":["/Users/wriitc/.pyenv/versions/Roguelike/bin/python","-m","ipykernel","-f","{connection_file}"],"display_name":"Roguelike","env":{}},"resource_dir":"/Users/wriitc/.ipython/kernels/Roguelike"}}}
