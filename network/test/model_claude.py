import tensorflow as tf
import scipy.sparse
import numpy as np
import os, time, collections, shutil, sys
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense

class BaseModel(Model):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.regularizers = []
        self.checkpoints = 'final'
        
    def predict(self, data, labels=None):
        """Prediction method that handles batching"""
        size = data.shape[0]
        predictions = np.empty((size, self.out_joints*3))
        loss = 0
        
        for begin in range(0, size, self.batch_size):
            end = min(begin + self.batch_size, size)
            
            batch_data = np.zeros((self.batch_size,) + data.shape[1:])
            tmp_data = data[begin:end]
            if isinstance(tmp_data, scipy.sparse.spmatrix):
                tmp_data = tmp_data.toarray()
            batch_data[:end-begin] = tmp_data
            
            batch_pred = self(batch_data, training=False)
            
            if labels is not None:
                batch_labels = np.zeros((self.batch_size,) + labels.shape[1:])
                batch_labels[:end-begin] = labels[begin:end]
                batch_loss = self.compute_loss(batch_pred[:end-begin], batch_labels[:end-begin])
                loss += batch_loss * (end - begin)
            
            predictions[begin:end] = batch_pred[:end-begin]
            
        if labels is not None:
            return predictions, loss / size
        return predictions

    def evaluate(self, data, labels):
        """Evaluation method"""
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels)
        
        string = 'loss: {:.4e}'.format(loss)
        string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(
            time.process_time()-t_process, time.time()-t_wall)
        return string, loss

    def fit(self, train_data, train_labels, val_data, val_labels):
        """Custom training loop with proper checkpoint handling"""
        t_process, t_wall = time.process_time(), time.time()
        
        # Setup checkpoint system
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        checkpoint_dir = self._get_path('checkpoints')
        
        # Create checkpoint managers for both final and best models
        self.ckpt = tf.train.Checkpoint(model=self, step=tf.Variable(0))
        final_manager = tf.train.CheckpointManager(
            self.ckpt, 
            os.path.join(checkpoint_dir, 'final'),
            max_to_keep=1
        )
        best_manager = tf.train.CheckpointManager(
            self.ckpt, 
            os.path.join(checkpoint_dir, 'best'),
            max_to_keep=1
        )
        
        # Setup for training
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        print(f"Total steps: {num_steps}")
        
        # Create optimizer with learning rate schedule
        if hasattr(self, 'decay_type') and hasattr(self, 'decay_params'):
            lr_schedule = self._create_lr_schedule(num_steps)
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        else:
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate)
        
        # Training loop
        min_loss = float('inf')
        losses = []
        indices = collections.deque()
        
        for step in range(1, num_steps + 1):
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]
            
            batch_data = train_data[idx]
            batch_labels = train_labels[idx]
            
            # Training step
            with tf.GradientTape() as tape:
                predictions = self(batch_data, training=True)
                loss = self.compute_loss(predictions, batch_labels)
            
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            
            # Update checkpoint step
            self.ckpt.step.assign_add(1)
            
            # Periodic evaluation
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print(f'step {step} / {num_steps} (epoch {epoch:.2f} / {self.num_epochs}):')
                print(f'  learning_rate = {optimizer.learning_rate.numpy():.2e}, loss = {loss:.4e}')
                
                val_string, val_loss = self.evaluate(val_data, val_labels)
                losses.append(val_loss)
                print(f'  validation {val_string}')
                
                # Save checkpoints
                final_manager.save()
                if val_loss < min_loss:
                    min_loss = val_loss
                    best_manager.save()
        
        print(f'validation loss: best = {min_loss:.4f}, final mean = {np.mean(losses[-10:]):.2f}')
        return losses, (time.time() - t_wall) / num_steps

    def _get_session(self, sess=None):
        """Get session and restore checkpoint"""
        if sess is None:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)
            
            # Restore checkpoint
            checkpoint_dir = os.path.join(self._get_path('checkpoints'), self.checkpoints)
            self.ckpt = tf.train.Checkpoint(model=self)
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print('restore from %s' % latest_checkpoint)
                self.ckpt.restore(latest_checkpoint)
            
        return sess

    def _create_lr_schedule(self, num_steps):
        """Create learning rate schedule based on decay parameters"""
        if self.decay_type == 'exp':
            return tf.keras.optimizers.schedules.ExponentialDecay(
                self.learning_rate,
                decay_steps=self.decay_params['decay_steps'],
                decay_rate=self.decay_params['decay_rate'],
                staircase=False)
        elif self.decay_type == 'step':
            return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.decay_params['boundaries'],
                self.decay_params['lr_values'])
        else:
            return self.learning_rate

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', 'experiment', self.dir_name, folder)

# Rest of the cgcnn class implementation remains the same as before
class cgcnn(BaseModel):
    # ... (rest of the implementation stays identical)
    """CGCNN model that maintains compatibility with the original interface"""

    def __init__(
        self,
        F=64,
        mask_type="locally_connected",
        init_type="ones",
        neighbour_matrix=None,
        in_joints=17,
        out_joints=17,
        in_F=2,
        num_layers=2,
        residual=True,
        batch_norm=True,
        max_norm=True,
        num_epochs=200,
        learning_rate=0.001,
        decay_type="exp",
        decay_params=None,
        regularization=0.0,
        dropout=0,
        batch_size=200,
        eval_frequency=200,
        dir_name="",
        checkpoints="final",
    ):
        super(cgcnn, self).__init__()

        # Store configuration
        self.F = F
        self.mask_type = mask_type
        self.init_type = init_type
        self.neighbour_matrix = neighbour_matrix
        self.in_joints = in_joints
        self.out_joints = out_joints
        self.in_F = in_F
        self.num_layers = num_layers
        self.residual = residual
        self.batch_norm = batch_norm
        self.max_norm = max_norm
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_type = decay_type
        self.decay_params = decay_params
        self.regularization = regularization
        self.dropout = dropout
        self.batch_size = batch_size
        self.checkpoints = checkpoints
        self.eval_frequency = eval_frequency
        self.dir_name = dir_name

        # Initialize mask
        self.initialize_mask()

        # Build model layers
        self.dense_layers = []
        self.bn_layers = []

        # Input layer
        self.input_dense = Dense(self.in_joints * self.F, use_bias=True)
        self.input_bn = BatchNormalization() if batch_norm else None

        # Hidden layers
        for i in range(num_layers):
            self.dense_layers.append(
                [
                    Dense(self.in_joints * self.F, use_bias=True),
                    Dense(self.in_joints * self.F, use_bias=True),
                ]
            )
            if batch_norm:
                self.bn_layers.append([BatchNormalization(), BatchNormalization()])

        # Output layer
        self.output_dense = Dense(self.out_joints * 3, use_bias=True)

    def initialize_mask(self):
        """Initialize the connectivity mask"""
        if "locally_connected" in self.mask_type:
            L = self.neighbour_matrix.T
            if "learnable" not in self.mask_type:
                self.mask = tf.constant(L, dtype=tf.float32)
            else:
                if self.init_type == "same":
                    mask_init = L
                elif self.init_type == "ones":
                    mask_init = np.ones_like(L)
                else:  # random
                    mask_init = np.random.uniform(0, 1, L.shape)

                self.mask = tf.Variable(
                    initial_value=mask_init,
                    dtype=tf.float32,
                    trainable=True,
                    name="mask",
                )
                self.mask.assign(
                    tf.nn.softmax(self.mask, axis=0) * tf.cast(L != 0, tf.float32)
                )

    def mask_weights(self, weights, input_size, output_size):
        """Apply connectivity mask to weights"""
        weights = tf.reshape(weights, [self.in_joints, -1, self.in_joints, -1])
        mask = tf.reshape(self.mask, [self.in_joints, 1, self.in_joints, 1])
        return tf.reshape(weights * mask, [input_size, output_size])

    def call(self, inputs, training=False):
        """Forward pass of the model"""
        x = tf.cast(inputs, tf.float32)

        # Input layer
        y = self.input_dense(x)
        if self.batch_norm:
            y = self.input_bn(y, training=training)
        y = tf.nn.leaky_relu(y)
        if training:
            y = tf.nn.dropout(y, self.dropout)

        # Hidden layers
        for i in range(self.num_layers):
            input_y = y

            # First dense
            y = self.dense_layers[i][0](y)
            if self.batch_norm:
                y = self.bn_layers[i][0](y, training=training)
            y = tf.nn.leaky_relu(y)
            if training:
                y = tf.nn.dropout(y, self.dropout)

            # Second dense
            y = self.dense_layers[i][1](y)
            if self.batch_norm:
                y = self.bn_layers[i][1](y, training=training)
            y = tf.nn.leaky_relu(y)
            if training:
                y = tf.nn.dropout(y, self.dropout)

            # Residual connection
            if self.residual:
                y = y + input_y

        # Output layer
        y = self.output_dense(y)

        # Reshape and combine with input
        x = tf.reshape(x, [-1, self.in_joints, self.in_F])
        y = tf.reshape(y, [-1, self.out_joints, 3])
        y = tf.concat(
            [x[:, :, :2] + y[:, :, :2], tf.expand_dims(y[:, :, 2], axis=-1)], axis=2
        )

        return tf.reshape(y, [-1, self.out_joints * 3])

    def compute_loss(self, predictions, labels):
        """Compute the loss function"""
        mse_loss = tf.reduce_mean(tf.square(predictions - labels))
        if self.regularization != 0:
            reg_loss = self.regularization * tf.add_n(
                [
                    tf.nn.l2_loss(v)
                    for v in self.trainable_variables
                    if "kernel" in v.name or "weight" in v.name
                ]
            )
            return mse_loss + reg_loss
        return mse_loss
