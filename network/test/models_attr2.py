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
        self.checkpoints = "final"

    def predict(self, data, labels=None):
        """Prediction method that handles batching"""
        size = data.shape[0]
        predictions = np.empty((size, self.out_joints * 3))
        loss = 0

        for begin in range(0, size, self.batch_size):
            end = min(begin + self.batch_size, size)

            batch_data = np.zeros((self.batch_size,) + data.shape[1:])
            # feed_dict REMOVED HERE
            tmp_data = data[begin:end]
            if isinstance(tmp_data, scipy.sparse.spmatrix):
                tmp_data = tmp_data.toarray()
            batch_data[: end - begin] = tmp_data

            #batch_pred = self(batch_data, training=False) non so cosa faccia

            if labels is not None:
                batch_labels = np.zeros((self.batch_size,) + labels.shape[1:])
                batch_labels[: end - begin] = labels[begin:end]
                batch_pred, batch_loss = self.compute_loss(
                    batch_pred[: end - begin], batch_labels[: end - begin]
                )
                loss += batch_loss * (end - begin)

            predictions[begin:end] = batch_pred[: end - begin]

        if labels is not None:
            return predictions, loss / size
        return predictions

    def evaluate(self, data, labels):
        """Evaluation method"""  # EVERYTHING IS OKAY HERE
        t_process, t_wall = time.process_time(), time.time()
        predictions, loss = self.predict(data, labels)

        string = "loss: {:.4e}".format(loss)
        string += "\ntime: {:.0f}s (wall {:.0f}s)".format(
            time.process_time() - t_process, time.time() - t_wall
        )
        return string, loss

    def fit(self, train_data, train_labels, val_data, val_labels):
        """Custom training loop with proper checkpoint handling"""
        t_process, t_wall = time.process_time(), time.time()

        # Setup checkpoint system
        shutil.rmtree(self._get_path("checkpoints"), ignore_errors=True)
        os.makedirs(self._get_path("checkpoints"))
        checkpoint_dir = self._get_path("checkpoints")

        # Create checkpoint managers for both final and best models
        self.ckpt = tf.train.Checkpoint(model=self, step=tf.Variable(0))
        final_manager = tf.train.CheckpointManager(
            self.ckpt, os.path.join(checkpoint_dir, "final"), max_to_keep=1
        )
        best_manager = tf.train.CheckpointManager(
            self.ckpt, os.path.join(checkpoint_dir, "best"), max_to_keep=1
        )

        # Tensorboard writer
        train_summary_writer = tf.summary.create_file_writer(
            os.path.join("./", "train")
        )
        val_summary_writer = tf.summary.create_file_writer(
            os.path.join("./", "validation")
        )

        # Setup for training
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        print(f"Total steps: {num_steps}")

        # Create optimizer with learning rate schedule 
        if hasattr(self, "decay_type") and hasattr(self, "decay_params"):
            lr_schedule = self._create_lr_schedule(num_steps)
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=lr_schedule
            )  # legacy for compatibility For m1/m2 models
        else:
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate
            )

        # Training loop
        min_loss = float("inf")
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
                loss, batch_loss = self.compute_loss(predictions, batch_labels)

                # Log training metrics
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", loss, step=step)
                    tf.summary.scalar(
                        "learning_rate", optimizer.learning_rate.numpy(), step=step
                    )

                    # Log weights and gradients histograms
                    if step % self.eval_frequency == 0:
                        for var in self.trainable_variables:
                            tf.summary.histogram(var.name, var, step=step)

            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))

            if step % self.eval_frequency == 0:
                with train_summary_writer.as_default():
                    for grad, var in zip(grads, self.trainable_variables):
                        if grad is not None:
                            tf.summary.histogram(
                                f"{var.name}/gradient", grad, step=step
                            )
            # Update checkpoint step
            self.ckpt.step.assign_add(1)

            # Periodic evaluation
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print(
                    f"step {step} / {num_steps} (epoch {epoch:.2f} / {self.num_epochs}):"
                )
                print(
                    f"learning_rate = {optimizer.learning_rate.numpy():.2e}, loss = {loss:.4e}"
                )

                val_string, val_loss = self.evaluate(val_data, val_labels)
                losses.append(val_loss)
                print(f"  validation {val_string}")

                with val_summary_writer.as_default():
                    tf.summary.scalar("validation/loss", val_loss, step=step)
                    for var in self.trainable_variables:
                        tf.summary.histogram(var.name, var, step=step)

                # Save checkpoints
                final_manager.save()
                if val_loss < min_loss:
                    min_loss = val_loss
                    best_manager.save()

        print(
            f"validation loss: best = {min_loss:.4f}, final mean = {np.mean(losses[-10:]):.2f}"
        )
        return losses, (time.time() - t_wall) / num_steps

    def _create_lr_schedule(self, num_steps):
        """Create learning rate schedule based on decay parameters"""
        if self.decay_type == "exp":
            return tf.keras.optimizers.schedules.ExponentialDecay(
                self.learning_rate,
                decay_steps=self.decay_params["decay_steps"],
                decay_rate=self.decay_params["decay_rate"],
                staircase=False,
            )
        elif self.decay_type == "step":
            return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.decay_params["boundaries"], self.decay_params["lr_values"]
            )
        else:
            return self.learning_rate

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, "..", "experiment", self.dir_name, folder)

    @DeprecationWarning
    def _get_session(self, sess=None):
        """Get session and restore checkpoint"""
        if sess is None:
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.compat.v1.Session(config=config)

            # Restore checkpoint
            checkpoint_dir = os.path.join(
                self._get_path("checkpoints"), self.checkpoints
            )
            self.ckpt = tf.train.Checkpoint(model=self)
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                print("restore from %s" % latest_checkpoint)
                self.ckpt.restore(latest_checkpoint)

        return sess

    @DeprecationWarning
    def _variable(self, name, initializer, shape, regularization=False):
        """Create a variable with optional regularization"""
        var = tf.Variable(initializer(shape, dtype=tf.float32), name=name)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        return var

    @DeprecationWarning
    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ":0")
        val = sess.run(var)
        sess.close()
        return val

    @DeprecationWarning
    def _variable(self, name, initializer, shape, regularization=True):
        var = tf.compat.v1.get_variable(
            name, shape, tf.float32, initializer=initializer, trainable=True
        )
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.compat.v1.summary.histogram(var.op.name, var)
        return var

    def build_graph(self, M_0, in_F):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Mask.
            self.initialize_mask()

            # Inputs.
            with tf.compat.v1.name_scope("inputs"):
                self.ph_data = tf.compat.v1.placeholder(
                    tf.float32, (self.batch_size, M_0 * in_F), "data"
                )
                self.ph_labels = tf.compat.v1.placeholder(
                    tf.float32, (self.batch_size, M_0 * 3), "labels"
                )
                self.ph_dropout = tf.compat.v1.placeholder(tf.float32, (), "dropout")
                self.ph_istraining = tf.compat.v1.placeholder(tf.bool, (), "istraining")

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_dropout)
            self.op_loss, self.op_loss_average = self.compute_loss(op_logits, self.ph_labels)
            self.op_train = self.training(
                self.op_loss, self.learning_rate, self.decay_type, self.decay_params
            )

            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.compat.v1.global_variables_initializer()
            
        self.graph.finalize()

# Questa classe è più o meno uguale. La differenza è che il metodo call che prima non esisteva.
class cgcnn(BaseModel):
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
        self.activation = tf.nn.leaky_relu

        # Build the computational graph.
        self.build_graph(in_joints, self.in_F)
        

    def compute_loss(self, predictions, labels):
        """Compute the loss function"""
        loss = 0
        mse_loss = tf.reduce_mean(tf.square(predictions - labels))
        if self.regularization != 0:
            reg_loss = self.regularization * tf.add_n(self.regularizers)
            loss *= reg_loss

        averages = tf.compat.v1.train.ExponentialMovingAverage(0.9)
        loss_average = tf.identity(averages.average(loss), name="loss_average")
        return mse_loss, loss_average
    
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
                    shape=(
                        (
                            [self.in_joints, self.in_joints]
                            if self.init_type != "same"
                            else None
                        ),
                    ),
                )
                self.mask.assign(
                    tf.nn.softmax(self.mask, axis=0) * tf.constant(L != 0, tf.float32)
                )

    def mask_weights(self, weights, input_size, output_size):
        """Apply connectivity mask to weights"""
        # DIFF
        input_size, output_size = weights.get_shape()
        input_size, output_size = int(input_size), int(output_size)
        assert input_size % self.in_joints == 0 and output_size % self.in_joints == 0
        in_F = int(input_size / self.in_joints)
        out_F = int(output_size / self.in_joints)

        weights = tf.reshape(weights, [self.in_joints, in_F, self.in_joints, out_F])
        mask = tf.reshape(self.mask, [self.in_joints, 1, self.in_joints, 1])
        return tf.reshape(weights * mask, [input_size, output_size])

    def batch_normalization_warp(self, y, training, name):
        keras_bn = tf.keras.layers.BatchNormalization(axis=-1, name=name)

        _, output_size = y.get_shape()
        output_size = int(output_size)
        out_F = int(output_size / self.in_joints)
        y = tf.reshape(y, [-1, self.in_joints, out_F])
        y = keras_bn(y, training=training)
        y = tf.reshape(y, [-1, output_size])

        for item in keras_bn.updates:
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, item)

        return y

    def kaiming(self, shape, dtype, partition_info=None):
        """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf

        Args
            shape: dimensions of the tf array to initialize
            dtype: data type of the array
            partition_info: (Optional) info about how the variable is partitioned.
                See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
                Needed to be used as an initializer.
        Returns
            Tensorflow array with initial weights
        """
        return tf.random.truncated_normal(shape, dtype=dtype) * tf.sqrt(
            2 / float(shape[0])
        )

    def _inference_lcn(self, x, data_dropout):

        with tf.compat.v1.variable_scope("linear_model"):

            mid_size = self.in_joints * self.F

            # === First layer===
            w1 = self._variable(
                "w1",
                self.kaiming,
                [self.in_joints * self.in_F, mid_size],
                regularization=self.regularization != 0,
            )
            b1 = self._variable(
                "b1", self.kaiming, [mid_size], regularization=self.regularization != 0
            )  # equal to b2leaky_relu
            w1 = tf.clip_by_norm(w1, 1) if self.max_norm else w1

            w1 = self.mask_weights(w1)
            y3 = tf.matmul(x, w1) + b1

            if self.batch_norm:
                y3 = self.batch_normalization_warp(
                    y3, training=self.ph_istraining, name="batch_normalization"
                )
            y3 = self.activation(y3)
            y3 = tf.nn.dropout(y3, rate=data_dropout)

            # === Create multiple bi-linear layers ===
            for idx in range(self.num_layers):
                y3 = self.two_linear(y3, data_dropout=data_dropout, idx=idx)

            # === Last layer ===
            input_size4 = int(y3.get_shape()[1])
            w4 = self._variable(
                "w4",
                self.kaiming,
                [input_size4, self.out_joints * 3],
                regularization=self.regularization != 0,
            )
            b4 = self._variable(
                "b4",
                self.kaiming,
                [self.out_joints * 3],
                regularization=self.regularization != 0,
            )
            w4 = tf.clip_by_norm(w4, 1) if self.max_norm else w4

            w4 = self.mask_weights(w4)
            y = tf.matmul(y3, w4) + b4
            # === End linear model ===

            x = tf.reshape(x, [-1, self.in_joints, self.in_F])  # [N, J, 3]
            y = tf.reshape(y, [-1, self.out_joints, 3])  # [N, J, 3]
            y = tf.concat(
                [x[:, :, :2] + y[:, :, :2], tf.expand_dims(y[:, :, 2], axis=-1)], axis=2
            )  # [N, J, 3]
            y = tf.reshape(y, [-1, self.out_joints * 3])

        return y
    

