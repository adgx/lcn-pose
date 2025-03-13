import tensorflow as tf

# Check GPU availability
print("GPU Available:", tf.test.is_gpu_available())

# List GPU devices
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Simple GPU computation test
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.compat.v1.Session() as sess:
    print("Multiplication result:", sess.run(c))