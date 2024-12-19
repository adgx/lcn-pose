import tensorflow as tf
import numpy as np

# Disable eager execution (TF 1.x style)
tf.compat.v1.disable_eager_execution()

# Set GPU memory growth to prevent allocation issues
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# Create a simple linear model
def create_linear_model():
    # Input placeholder
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='X')
    
    # Model parameters
    W = tf.Variable(tf.random.normal([1, 1]), name='weights')
    b = tf.Variable(tf.zeros([1]), name='bias')
    
    # Linear model operation
    with tf.device('/gpu:0'):
        Y_pred = tf.matmul(X, W) + b
    
    return X, Y_pred, W, b

# Generate some random data
np.random.seed(42)
X_data = np.random.rand(100, 1).astype(np.float32)
Y_data = 2 * X_data + 1 + np.random.normal(0, 0.05, (100, 1))

# Create the model
X, Y_pred, W, b = create_linear_model()

# Define loss and optimizer
Y_true = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
loss = tf.reduce_mean(input_tensor=tf.square(Y_true - Y_pred))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

# Create a session and run
try:
    with tf.compat.v1.Session(config=config) as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Training loop
        for i in range(100):
            _, loss_val = sess.run([optimizer, loss], 
                                   feed_dict={X: X_data, Y_true: Y_data})
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {loss_val}")
        
        # Print final weights
        final_weights = sess.run(W)
        print("Final Weights:", final_weights)

except Exception as e:
    print("Error occurred:")
    print(str(e))