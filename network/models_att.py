import tensorflow as tf
import scipy.sparse
import numpy as np
import os, time, collections, shutil, sys, re


ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(ROOT_PATH)

import math
import glob


def get_exponential_matrix():
    """Generate an exponential matrix for the graph.
    Returns:
        np.ndarray: A matrix where each element is exp(1/2^dist[i][j]).
    """
    edges = [(0, 1, 1),
            (0, 7, 1),
            (0, 4, 1),
            (4, 5, 1),
            (4, 7, 1),
            (5, 6, 1),
            (1, 7, 1),
            (1, 2, 1),
            (2, 3, 1),
            (7, 11, 1),
            (7, 8, 1),
            (7, 14, 1),
            (11, 12, 1),
            (12, 13, 1),
            (14, 15, 1),
            (15, 16, 1),
            (8, 9, 1),
            (9, 10, 1)]

    n_nodes = 17
    INF = np.inf

    #initialize
    dist = np.full((n_nodes, n_nodes), INF)
    np.fill_diagonal(dist, 0)

    # graph undirect
    for u, v, w in edges:
        dist[u][v] = w
        dist[v][u] = w 

    # Floyd–Warshall
    for k in range(n_nodes):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    print("Distances matrix:")
    print(dist)

    #exp_mat = np.exp(1/(2**(dist)))
    exp_mat = 1 / (2 ** dist)
    print("exp_mat:")
    print(exp_mat)

    #Convert to float32

    exp_mat = exp_mat.astype(np.float32)

    return exp_mat

class base_model(object):

    def __init__(self):
        self.regularizers = []
        self.checkpoints = "final"
        self.writer = None
    # High-level interface which runs the constructed computational graph.
    #inferece uses the test_data
    def predict(self, data, labels=None, sess=None):
        loss = 0
        size = data.shape[0] #n frames per videocamera
        predictions = np.empty((size, self.out_joints * 3)) #[size, joints(17)*3]
        #doto find more info for the close_sess_flag
        close_sess_flag = True if sess is None else False
        sess = self._get_session(sess)
        for begin in range(0, size, self.batch_size):
            # If the last batch is smaller than a usual batch, fill with zeros.
            end = begin + self.batch_size
            end = min([end, size]) # not overcome the size of the data 

            #create a numpy array where store the data for processing
            batch_data = np.zeros((self.batch_size,) + data.shape[1:]) #[batch_size, data.shape[1:]]
            #extract the batch from dataset
            tmp_data = data[begin:end]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[: end - begin] = tmp_data
            
            #disable droput, and state that model is not training 
            feed_dict = {
                self.ph_data: batch_data,
                self.ph_dropout: 0,
                self.ph_istraining: False,
            }

            #labels->projected joints coordinates in cam coordinates, and normalizated by the cam resolution 
            #Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros((self.batch_size,) + labels.shape[1:])
                batch_labels[: end - begin] = labels[begin:end]
                feed_dict[self.ph_labels] = batch_labels
                batch_pred, batch_loss = sess.run(
                    [self.op_prediction, self.op_loss], feed_dict
                )
                loss += batch_loss
            else:
                #op_prediction is the tensorflow operation that represents the 
                # model's predictions
                #So the tensorflow compute the predictions opreations based on 
                # the input data feed_dict
                batch_pred = sess.run(self.op_prediction, feed_dict)

            #store the predictions
            predictions[begin:end] = batch_pred[: end - begin]

        if close_sess_flag:
            sess.close()

        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions

    def evaluate(self, data, labels, sess=None):
        """Show the loss value"""
        t_process, t_wall = time.process_time(), time.time()
        #the loss value is 0 if there no labels
        predictions, loss = self.predict(data, labels, sess)
        
        string = "loss: {:.4e}".format(loss)

        if sess is None:
            string += "\ntime: {:.0f}s (wall {:.0f}s)".format(
                time.process_time() - t_process, time.time() - t_wall
            )
        return string, loss

    def fit(self, train_data, train_labels, val_data, val_labels, output_dir=None, starting_checkpoint=None):
        tf.compat.v1.disable_eager_execution() # Fix save model issue
        t_process, t_wall = time.process_time(), time.time()

        starting_step = 1

        #set the configurations
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        #make the session
        sess = tf.compat.v1.Session(graph=self.graph, config=config)

        #make dirs for the checkpoints
        path = os.path.join(self._get_path("checkpoints"), "final", "model")
        best_path = os.path.join(self._get_path("checkpoints"), "best", "model")
        #check there is a checkpoint from start
        if starting_checkpoint is None:
            shutil.rmtree(self._get_path("checkpoints"), ignore_errors=True)
            os.makedirs(self._get_path("checkpoints"))
        else:
            #Get the parent directory of the checkpoint
            #In this way is condsider always model-5.index (this make sense?)
            match = re.search(r'model-(\d+)', starting_checkpoint + "/model-5.index")
            if starting_checkpoint and tf.io.gfile.exists(starting_checkpoint + "/model-5.index"):
                if match:
                    starting_step = int(match.group(1))
                    print(f"Resuming from step {starting_step}")
                    print(f"Restoring from checkpoint: {starting_checkpoint}")
                    self.op_saver.restore(sess, tf.train.latest_checkpoint(starting_checkpoint))

        #compute a step
        sess.run(self.op_init)
        # Training.
        losses = []
        #queue double ended
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size) # Numero di step totali già moltiplicati per epoche
        eval_frequency = num_steps // self.num_epochs 
        #epoch_steps = int(train_data.shape[0] / self.batch_size)
        print(f"Total steps to be done to complete all the epochs: {num_steps}")
        min_loss = 10000
        training_error = []
        validation_error = []

        for step in range(starting_step, num_steps + 1):
            if len(indices) < self.batch_size:
                #take a random premutation of the indices of the samples 
                indices.extend(np.random.permutation(train_data.shape[0]))
                #make a list of batch_size indices from the indeces deque
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx, ...], train_labels[idx, ...]
            feed_dict = {
                self.ph_data: batch_data,
                self.ph_labels: batch_labels,
                self.ph_dropout: self.dropout,
                self.ph_istraining: True,
            }
            #op_train defines the operations for the training.
            #It applies gradients to update the model's parameters 
            # based on the loss function and learning rate.
            learning_rate, loss_average = sess.run(
                [self.op_train, self.op_loss_average], feed_dict
            )

            # Periodical evaluation of the model.
            if step % eval_frequency == 0:
                epoch = step * self.batch_size / train_data.shape[0]
                print(
                        "step {} / {} (epoch {:.2f} / {}):".format(
                            step, num_steps, epoch, self.num_epochs
                        )
                    )
                    
                print(
                        "  learning_rate = {:.2e}, loss_average = {:.4e}".format(
                            learning_rate, loss_average
                        )
                    )
                training_error.append([
                    time.time(),
                    step,
                    loss_average,
                ])
                string, loss = self.evaluate(val_data, val_labels, sess)
                losses.append(loss)
                print("validation {}".format(string))
                print(
                    "time: {:.0f}s (wall {:.0f}s)".format(
                        time.process_time() - t_process, time.time() - t_wall
                    )
                )
                validation_error.append([
                    0, # Valore utile solo per la compatibilità con la generazione dei dati da colab
                    step,
                    loss,
                ])

                # Summaries for TensorBoard.
                #protocol buffer to encapsulate data for visualization
                summary = tf.compat.v1.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag="validation/loss", simple_value=loss)
                #save the summary in the file event
                self.writer.add_summary(summary, step)

                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)
                #Save the best checkpoint
                if loss < min_loss:
                    min_loss = loss
                    self.op_best_saver.save(sess, best_path, global_step=step)

        print(
            "validation loss: trough = {:.4f}, mean = {:.2f}".format(
                min_loss, np.mean(losses[-10:])
            )
        )
        self.writer.close()
        sess.close()

        #time per step
        t_step = (time.time() - t_wall) / num_steps

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            import json
            with open(output_dir + "/training_error.json", "w") as f:
                f.write(json.dumps([training_error], indent=4))
                  
            with open(output_dir + "/validation_error.json", "w") as f:
                f.write(json.dumps([validation_error], indent=4))

        #Print the final training and validation errors
        print("Final training error: ", training_error[-1])
        print("Final validation error: ", validation_error[-1])
        return losses, t_step

    def build_graph(self, M_0, in_F):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        #setting the computational graph
        with self.graph.as_default():
            # Mask.
            self.initialize_mask()

            #make the Inputs scope
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
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels)
            self.op_train = self.training(
                self.op_loss, self.learning_rate, self.decay_type, self.decay_params
            )
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.compat.v1.global_variables_initializer()

            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.compat.v1.summary.merge_all()
            self.op_saver = tf.compat.v1.train.Saver(max_to_keep=1)
            self.op_best_saver = tf.compat.v1.train.Saver(max_to_keep=1)

            # Writer for TensorBoard.
            # Remove the previous logs
            if os.path.exists(self._get_path("summaries")):
                shutil.rmtree(self._get_path("summaries"))
            
            self.writer = tf.compat.v1.summary.FileWriter(
                self._get_path("summaries"), self.graph
            )
            
        self.graph.finalize()

    def initialize_mask(self):
        self._initialize_mask()

    def inference(self, data, dropout):
        logits = self._inference_lcn(data, data_dropout=dropout)
        return logits

    def probabilities(self, logits):
        with tf.compat.v1.name_scope("probabilities"):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        with tf.compat.v1.name_scope("prediction"):
            prediction = tf.compat.v1.identity(logits)
            return prediction

    def loss(self, logits, labels):
        with tf.compat.v1.name_scope("loss"):
            loss = 0
            with tf.compat.v1.name_scope("mse_loss"):
                mse_loss = tf.reduce_mean(input_tensor=tf.square(logits - labels))
                # logits = tf.reshape(logits, [-1, self.out_joints, 3])
                # labels = tf.reshape(labels, [-1, self.out_joints, 3])
                # mse_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(logits - labels), axis=2)))
            loss = loss + mse_loss

            if self.regularization != 0 and self.regularization is not None:
                with tf.compat.v1.name_scope("reg_loss"):
                    reg_loss = self.regularization * tf.add_n(self.regularizers)
                loss += reg_loss

            # Summaries for TensorBoard.
            tf.compat.v1.summary.scalar("loss/mse_loss", mse_loss)
            tf.compat.v1.summary.scalar("loss/total", loss)
            with tf.compat.v1.name_scope("averages"):
                averages = tf.compat.v1.train.ExponentialMovingAverage(0.9)
                loss_dict = {"mse": mse_loss, "total": loss}
                op_averages = averages.apply(list(loss_dict.values()))
                for k, v in loss_dict.items():
                    tf.compat.v1.summary.scalar("loss/avg/%s" % k, averages.average(v))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.compat.v1.identity(
                        averages.average(loss), name="control"
                    )
            return loss, loss_average

    def training(self, loss, learning_rate, decay_type, decay_params):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.compat.v1.name_scope("training"):
            # Learning rate.
            global_step = tf.compat.v1.get_variable(
                name="global_step",
                dtype=tf.int32,
                trainable=False,
                initializer=0,
            )
            if decay_type == "exp":
                learning_rate = tf.compat.v1.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_params["decay_steps"],
                    decay_params["decay_rate"],
                    staircase=False,
                )
            else:
                assert 0, "not implemented lr decay types!"
            tf.compat.v1.summary.scalar("learning_rate", learning_rate)
            # Optimizer.
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads = optimizer.compute_gradients(loss)
                op_gradients = optimizer.apply_gradients(grads, global_step=global_step)

            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print("warning: {} has no gradient".format(var.op.name))
                else:
                    tf.compat.v1.summary.histogram(var.op.name + "/gradients", grad)

            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.compat.v1.identity(learning_rate, name="control")
            return op_train

    # Helper methods.
    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ":0")
        val = sess.run(var)
        sess.close()
        return val

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, "..", "experiment", self.dir_name, folder)

    def remove_checkpoint(checkpoint_prefix):
        # Remove the checkpoint file
        if os.path.exists(checkpoint_prefix):
            os.remove(checkpoint_prefix)
        
        # Remove associated files (index, data, etc.)
        for file in glob.glob(checkpoint_prefix + ".*"):
            os.remove(file)
        
    #Restore parameters if no session given    
    def _get_session(self, sess=None):
        """
        Restore parameters if no session given.
        """
        if sess is None:
            tf.compat.v1.disable_eager_execution() # Fix save model issue
        
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            #create a session
            sess = tf.compat.v1.Session(graph=self.graph, config=config)
            print(self._get_path("checkpoints"))
            filename = tf.train.latest_checkpoint(
                os.path.join(self._get_path("checkpoints"), self.checkpoints)
            )
            print("restore from %s" % filename)
            #restore the model variables stored into the checkpoint
            self.op_best_saver.restore(sess, filename)
        return sess

    def _variable(self, name, initializer, shape, regularization=True):
        var = tf.compat.v1.get_variable(
            name, shape, tf.float32, initializer=initializer, trainable=True
        )
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))#sum(var**2)/2
        tf.compat.v1.summary.histogram(var.op.name, var)
        return var


class cgcnn(base_model):
    """ """
    #set all parameters for the model
    def __init__(
        self,
        F=64, 
        mask_type="locally_connected", #it can be locally_connected_learnable
        #we can try with different type of inizialization for hope a stroke of lucky for the lernable parameter for S matrix
        init_type="ones", #same: use L to init learnable part in mask
                          #ones: use 1 to init learnable part in mask
                          #random: use random to init learnable part in mask
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
        is_training=True,
        knn=1 # not used
    ):
        super().__init__()

        self.F = F
        self.mask_type = mask_type 
        self.init_type = init_type
        assert neighbour_matrix.shape[0] == neighbour_matrix.shape[1]
        assert neighbour_matrix.shape[0] == in_joints
        self.neighbour_matrix = neighbour_matrix
        #what is the neighbour matrix: see the paper section 3
        self.in_joints = in_joints
        self.out_joints = out_joints
        self.num_layers = num_layers
        self.residual, self.batch_norm, self.max_norm = residual, batch_norm, max_norm
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_type, self.decay_params = decay_type, decay_params
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.checkpoints = checkpoints
        self.activation = tf.nn.leaky_relu
        self.in_F = in_F
        self.is_training = is_training
        self.knn = knn

        # Build the computational graph.
        self.build_graph(in_joints, self.in_F)

    def _initialize_mask(self):
        """
        Parameter
            mask_type
                exponential
                locally_connected
                locally_connected_learnable
            init_type
                same: use L to init learnable part in mask
                ones: use 1 to init learnable part in mask
                random: use random to init learnable part in mask
        """
        #can mask_type have more values? yes see learnable
        if "locally_connected" in self.mask_type:
            assert self.neighbour_matrix is not None
            L = self.neighbour_matrix.T #[17, 17] transpose matrix
            assert L.shape == (self.in_joints, self.in_joints)

            if self.init_type == "same":
                initializer = L
            else:
                raise ValueError("Unknown init_type: {}".format(self.init_type))

            var_mask = tf.compat.v1.get_variable(
                    name="mask",
                    shape=(
                        [self.in_joints, self.out_joints]
                        if self.init_type != "same"
                        else None
                    ),
                    dtype=tf.float32,
                    initializer=initializer, #set the value for the mask
                )
                #applay the softmax to the matrix, for have the probability
                #what is the relationship between the in_join and out_join?
            var_mask = tf.nn.softmax(var_mask, axis=0)
            # self.mask = var_mask
            self.mask = var_mask * tf.constant(L != 0, dtype=tf.float32)
            
        if "exponential" in self.mask_type:
            self.mask = tf.constant(get_exponential_matrix())
    def mask_weights(self, weights):
        input_size, output_size = weights.get_shape() #[34, 1088]
        input_size, output_size = int(input_size), int(output_size)
        assert input_size % self.in_joints == 0 and output_size % self.in_joints == 0
        in_F = int(input_size / self.in_joints) #2
        out_F = int(output_size / self.in_joints) #64
        weights = tf.reshape(weights, [self.in_joints, in_F, self.in_joints, out_F])#[17,2,17,64]
        mask = tf.reshape(self.mask, [self.in_joints, 1, self.in_joints, 1])#[17, 1, 17, 1]
        masked_weights = weights * mask
        masked_weights = tf.reshape(masked_weights, [input_size, output_size]) #[34, 1088]
        return masked_weights
    
    def batch_normalization_warp(self, y, training, name):
        """
        Batch normalization wrapper for Keras layers.
        Args:
            y: input tensor to be normalized
            training: boolean, whether the model is in training mode
            name: name for the batch normalization layer
        Returns:
            y: normalized tensor
        """
        tf.compat.v1.disable_eager_execution()
        keras_bn = tf.keras.layers.BatchNormalization(axis=-1, name=name)

        _, output_size = y.get_shape()
        output_size = int(output_size)
        out_F = int(output_size / self.in_joints)
        y = tf.reshape(y, [-1, self.in_joints, out_F])
        
        y = keras_bn(y, training=training) 
        y = tf.reshape(y, [-1, output_size])

        #for item in keras_bn.updates:
        #    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, item)

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

    def two_linear(self, xin, data_dropout, idx):
        """
        Make a bi-linear block with optional residual connection

        Args
            xin: the batch that enters the block
            idx: integer. Number of layer (for naming/scoping)
            Returns
        y: the batch after it leaves the block
        """

        with tf.compat.v1.variable_scope("two_linear_" + str(idx)) as scope:

            output_size = self.in_joints * self.F

            # Linear 1
            input_size2 = int(xin.get_shape()[1])
            w2 = self._variable(
                "w2_" + str(idx),
                self.kaiming,  #array that contains the truncated values from a distribution with mean 0 and standard deviation = 1
                [input_size2, output_size],
                regularization=self.regularization != 0,
            )
            b2 = self._variable(
                "b2_" + str(idx),
                self.kaiming,
                [output_size],
                regularization=self.regularization != 0,
            )
            w2 = tf.clip_by_norm(w2, 1) if self.max_norm else w2
            w2 = self.mask_weights(w2)
            #linear transformation
            y = tf.matmul(xin, w2) + b2

            if self.batch_norm:
                y = self.batch_normalization_warp(
                    y,
                    training=self.is_training,
                    name="batch_normalization1" + str(idx),
                )
            #Leaky ReLU
            y = self.activation(y)
            #apply the dropout
            y = tf.nn.dropout(y, rate=data_dropout)
            # ====================

            # Linear 2
            input_size3 = int(y.get_shape()[1])
            w3 = self._variable(
                "w3_" + str(idx),
                self.kaiming,
                [input_size3, output_size],
                regularization=self.regularization != 0,
            )
            b3 = self._variable(
                "b3_" + str(idx),
                self.kaiming,
                [output_size],
                regularization=self.regularization != 0,
            )
            w3 = tf.clip_by_norm(w3, 1) if self.max_norm else w3
            w3 = self.mask_weights(w3)
            y = tf.matmul(y, w3) + b3
            if self.batch_norm:
                y = self.batch_normalization_warp(
                    y,
                    training=self.is_training,
                    name="batch_normalization2" + str(idx),
                )
            y = self.activation(y)
            y = tf.nn.dropout(y, rate=data_dropout)
            # ====================

            # Residual every 2 blocks
            y = (xin + y) if self.residual else y
        return y

    def _inference_lcn(self, x, data_dropout):
        #define the variables for the linear_model scope
        with tf.compat.v1.variable_scope("linear_model"):

            mid_size = self.in_joints * self.F #[1088=17*64]

            # === First layer===
            w1 = self._variable(
                "w1",
                self.kaiming,
                [self.in_joints * self.in_F, mid_size], #[34, 1088]
                regularization=self.regularization != 0,
            )

            b1 = self._variable(
                "b1", self.kaiming, [mid_size], regularization=self.regularization != 0
            )  # equal to b2leaky_relu
            #max_norm = True (default)
            #clip the values of w1 so that the L2-norm is less or equal to 1
            w1 = tf.clip_by_norm(w1, 1) if self.max_norm else w1

            w1 = self.mask_weights(w1) #[34, 1088]
            y3 = tf.matmul(x, w1) + b1

            #default true
            if self.batch_norm:
                #It normalizes the inputs to a layer by adjusting and scaling 
                # the activations to have a mean of 0 and a standard deviation 
                # of 1 for each batch.
                y3 = self.batch_normalization_warp(
                    y3, training=self.is_training, name="batch_normalization"
                )
            #apply the ReLU
            y3 = self.activation(y3)
            #apply the dropout  
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

            x = tf.reshape(x, [-1, self.in_joints, self.in_F])  # [N, J, 2]
            y = tf.reshape(y, [-1, self.out_joints, 3])  # [N, J, 3]
            y = tf.concat(
                [x[:, :, :2] + y[:, :, :2], tf.expand_dims(y[:, :, 2], axis=-1)], axis=2
            )  # [N, J, 3]
            y = tf.reshape(y, [-1, self.out_joints * 3])

        return y
    
#notes:
#GCN(Graph Convolutional Network) computes the output features of a node only depend on the nodes which are regarded as related 
# determined by the Laplacian matrix.
#In GCN the Laplacian operator is obtained as the product of structure matrix which encodes the dependence relation among the 
# nodes,and a weigth matrix which defines how to aggregate the dependent features.
#The weight matrix has an inherent weight sharing scheme, the learnable operators T are the same for all nodes.
#The structure matrix is directly determined by node distance (no customized node dependence).
#
#Locally Connected Network  discards the weight sharing scheme by freeing all of the parameters in the weight matrix, so 
# the nodes have their own operators. The structure matrix is constructed according to human joint dependence. 
#Laplacian operator into the product of a structure matrix and a weight matrix.
#
#Revisit GCN
#
#Features defined on a graph G = (V, E, W) where V rappresent a set of N nodes, E the edges e W a weighted adjacency
# matrix.
# x into R^N is a feature defined on the N nodes where each dimension correspnds to one node.  
#There are M features in total for each node. So is used the matrix X into R^M*N.(Row: feature m, col: feature m for the node n)
#Xr (X(m, :) the mth feature of all nodes) and Xc (X(:, n) the fatures of the nth node) to denote the flattened copies of X in row and column major order.
#The combinatorial definition of the Graph Laplacian L is computed as : L = D - A into R^N*N
# where D is the degree matrix and A is the adjacency matrix.
#The Laplacian can be diagonalized by the Fourier basis U into R^N*N that is the matrix of eigenvectors of the 
# Graph Laplacian so L = U*Λ*U^T.
#Where Λ represents a diagonal matrix that has eigenvalues of the Laplacian matrix, that provides structural information 
# about the graph in terms of connectivity, diffusion, and frequency modes.
#The graph Fourier transform of a feature vector x into R^N is:
#y=gθ(L)*x=U*gθ(Λ)*U^T*x
#
#gθ(Λ) = sum(k=0, K-1)θk*Λ^k, where K is a hyperparameter, and θ is a learnable parameter.  
#
#Reformulation
#
#To obtain an output feature vector y into R^N for N nodes by applying a filter gθ to the input features X
#
#y = gθ(X) = sum(k=0, K-1)sum(m=1, M) θkm * L^k * X(m,:)^T
#
#The fielter has different θ for different feature dimensions. However, diffent nodes in the graph share the same 
# filter θ. Same set of θ's is used for computing different dimensions of y which correspond to different nodes.
#
#Evaluating the output feature corresponding to the qth node yq, which is the qth dimension of y 
#
#yq = sum(k=0, K-1)sum(m=1, M)θkm * L^k(q, :) * X(m, :)^T
#
#Based on the characteristics of the Laplacian matrix L^k, if the minimum number of edges connectiong the joints
# i and j is larger than k, then L^k(i, j) = 0. So this can be interpreted as aggregation features from the 
# neighboring nodes whose distance is less than K.
#So:
#
#yq = sum(k=0, K-1)sum(m=1, M)X(m, :)*L^k(q, :)^T*θkm =sum(k=0, K-1)sum(m=1, M)X(m, :)*(L^k(q:)T H-mul Θkm)  
# where Θkm into R^Nx1 with θkm repeated N times. 
#
#Concatenation means stacking vectors or matrices along a given axis, so it arranges the elements in a sequence.
#Finally:
#
#yq = sum(k = 0, K-1)Xr*(Skq H-mul Wkq)
#where skq into R^MNx1 is a vector with L^k(:, q) repeated N times. It constains the neighbourhood information of node q.
#For all nodes in the graph
#
#y = sum(k=0, K-1) Xr * (S^k H-mul W^k)
#where both S^k and W^k are 2D matrices with the shape of MNxN. Skq it the qth column of S^k and Wkq is the qth column 
# of W^k.
#
#Limitations 
#
#The main limitation lies in the weight shariang scheme in W^k. Wkq has only M unique parameters because each Θkm 
# has unique parameter. Recall Θkm is obtained by repeating θkm N times. Another, GCN computes features for different
# nodes usign the same set of parameters => Wkq = Wkp.
#The way GCN constructs the structure matrix S treats all neighbors of the same distance to the node of interest without 
# discrimination, leaving us no flexibility to freely connect the joints of arbitrary distance.
#
#Generalization
#
#Obtain a more generic model by dropping the structure constraints in S^k and W^k. Is been developed an approach to 
# construct the structure matrix S to directly reflect the joint dependence.
#
#y = X*(S H-mul W)
#
#Relation to FCN and GCN
#
#FCN is essentially represented by the product of a weight matrix and a feature matrix. Customizing the generic model into 
# FCN can be achieved by settign all values in S to be 1
#
#y = X*(1 H-mul W)
#
#which means all of the nodes are connected. The parameters in W are all free, and are learned end to end form traning datasets.
#GCN can be obtained by initializing S^k and W^k and stacking S^k, W^k, (k=0, ..., K-1) vertically to generate S and W. 
#
#Locally Connected Network
#
#LCN is also a specialization of the generic model. But there are no constraints in W, and the joints are sparsely connected 
# in S.
#S is shared for all LCN layers which is offline constructed based on the specified joint dependence. Different layers
# have their own W which is learned end-to-end.
#
#Joint Dependence
#Joint dependence is locality which means each joint only depends on those which have short manifold distance to it.
# The mainifold distance between two joints is defined by their distance on the graph.
#So each joint depends on the neighbors whose manifold distance to the joint is less equal than K.
#This approach is called LCN(K-NN). K is used to wvaluate the 3D estimation accurancy.
#
#Structure Matrix
#
#S matrix is constructed to reflect the joint dependence. If joint j is dependent on the joint i, then is setted 
# the (i, j) block of S to be ones. Otherwise, we set it to be zeros.
#In this way, the features of the ith node will not contribute to the computation of the output features of the jth node.
#
#The operation in LCN:
#
#u^j = sum(i=1, N)h^i*(S^(i, j) H-mul W^(i, j))
#Where h^i denotes the M input features of the ith joint -> h^i = X(:, i)^T into R1xM.
#u^j into R^1xM' deontes the M' output feature of the node j.
#So if S(i, j) is zero, then h(i) will not contribute to the computation of u(j).
#One LCN layer gernerates output features for all nodes.
#The ones in S matrix are replaced by continuous values to reflect the importance among the joints. 
#We can also learn them from the training dataset end-to-end by replacing the non zero values in S with learnable parameters.
#
#Application to 3D Pose Estimation
#
#For the 3D pose estimation, is used a deep neural network which uses the LCN layer as the basic building block.
#LCN has several cascaded blocks with each consisting of two LCN layers, interleaved with BN, LeakyReLU and Dropout.
#Each block is wrapped in a residual connection.
#The number of output features M' in each LCN layer is set to be 64.
#The layers have different weight matrices.
#The input to the LCN network is the 2D locations of the body joints and the output is the corrensponding 3D locations.
#
#Experiments
#
#Datasets and Metrics
#
#It is computed the Mean Per Joint Position Error (MPJPE) between the ground truth and the 3D pose estimation after aligning
# the mid-hip joints. Also computes the estiamtions are aligned eith the ground truth via a rigid transformation.
#Others metrics of average used are PCK and AUC. 
#
#Implementation Details 
#
#Coordinate system
#
#Pw is the 3D location of a joint in the world Coordinate System (CS)
#Next transform the Pw to Camera coordinate system Pc = R*(Pw - T)
# where R is the rotation matrix and T is the translation vector.
#Then projects the Pc=(Xc, Yc, Zc) to 2D image plane
#Pp = (fx*Xc/Zc + cx, fy*Yc/Zc + cy)
# where fx,fy are focal lengths, and cx, cy are principal point coordinates.
#Pc is influnced by pose scale so to remove the scale in Pc is seeked a scalar λ which makes λPc have similar scale as 
# Pp by minimizing ||λP(cap)c -P(cap)p||2 where P(cap)c and P(cap)p denote the poses centered around their pelvis joints.
#So is estimated λPc form Pp which is independent of the actual body scale.
#
#2D detections 
#The input of the network is 2D poses estimated by the Stacked Hourglass.
#
#Baselines
#Baselines: FCN variant, GCN variant for 3D pose estimation, LCN (K-NN) where K ranges from 1 to 4, LCN (K-NN)-Learn 
# where the blocks of ones in S are repalce by learnable parameters to reflect its degree of dependence on the other joints.
#Finnaly the LCN-Learn where S is completely learn it from data.
#
#Notes:
#Laplacian operator: it is derived from the graph Laplacian matrix, which captures the structure of the graph and how information
# propagates across nodes.
#Graph Laplacian L is computed as : L = D - A where D is the degree matrix and A is the adjacency matrix,
# L is a discete version of the Laplacian operator that is used in continuous domanin.
#Normalized Laplacian:
#Lsyn = I - D^-1/2 * A * D^-1/2
#Λ
#θ
#Θ
#λ
#Hadamard (element-wise) multiplication
#
#While ReLU outputs zero for all negative input values, potentially leading to inactive neurons—a phenomenon known as the "dying ReLU" 
# problem—Leaky ReLU introduces a small, non-zero gradient for negative inputs, allowing the network to learn from them and mitigating 
# the risk of neuron inactivity
#
#Residual connections allows the input of a layer to bypass one or more intermediate layers and be added directly to the output 
# of a subsequent layer.gradients can become extremely small during backpropagation, hindering effective training. Residual 
# connections provide shortcut paths for gradient flow, alleviating this issue
#
#Dropout is to randomly deactivate a subset of neurons during each training iteration. This means that in every forward pass
#  through the network, certain neurons are "dropped out" or ignored, along with their connections.
#
#Extrinsic camera parameters define the position and orientation of a camera in the world coordinate system. They consist of two components:​
#-Rotation Matrix (R): A 3×3 orthogonal matrix that represents the orientation of the camera relative to the world coordinate system. 
# It aligns the axes of the world coordinate system with those of the camera's coordinate system. ​
#-Translation Vector (T): A 3×1 vector that specifies the position of the camera's origin in the world coordinate system. 
# It denotes the displacement of the camera from the world's origin. ​
#
#Intrinsic camera parameters are fundamental characteristics that define how a camera captures and projects a 3D scene onto a 2D image 
# plane.
#They are typically represented in a 3×3 matrix known as the intrinsic matrix (K), which facilitates the transformation from camera 
# coordinates to pixel coordinates in the image.
#fx​ and fy​: These represent the focal lengths in pixels along the x and y axes, respectively. They are calculated by multiplying the 
# physical focal length of the camera lens by the pixel density (number of pixels per unit length) along each axis. ​
#cx​ and cy​: These denote the coordinates of the principal point (also known as the optical center) in the image plane, typically 
# measured in pixels. Ideally, this point is located at the center of the image sensor. ​
#γ: This is the skew coefficient, which accounts for any non-orthogonality between the x and y pixel axes.

#Il parametro K è usato per la definizione della matrice di struttura S che va ad indicare la dipendenza di ciascun joint dai suoi 
# vicini, se la lunghezza del cammino tra un joint e l'atro è maggiore di k allora i due joint non saranno dipendenti.
# Quindi risulterebbe inutile avere K > 3 

#Potremmo usare un numero minore di azioni su cui allenare il modello in modo da verificare le sue capacita di generalizzazione

#usare meno dati possibili sulla base delle azioni e vedere chi riesce a fornire un risultato migliore in modo da far risaltare le 
# capacità di generalizzazione

#Prendere come paragone una rete allo stato d'arte attuale