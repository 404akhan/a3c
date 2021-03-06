import numpy as np
import tensorflow as tf

class Model():
  def __init__(self, num_outputs):
    self.num_outputs = num_outputs

    self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name='X')
    self.targets_pi = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
    self.targets_v = tf.placeholder(shape=[None], dtype=tf.float32, name="y") 
    self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

    X = tf.to_float(self.states) / 255.0
    batch_size = tf.shape(self.states)[0]

    conv1 = tf.contrib.layers.conv2d(
      X, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
    conv2 = tf.contrib.layers.conv2d(
      conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")

    # Fully connected layer
    fc1 = tf.contrib.layers.fully_connected(
      inputs=tf.contrib.layers.flatten(conv2),
      num_outputs=256,
      scope="fc1")

    ### Policy
    self.logits_pi = tf.contrib.layers.fully_connected(fc1, num_outputs, activation_fn=None)
    self.probs_pi = tf.nn.softmax(self.logits_pi) + 1e-8

    self.predictions_pi = {
      "logits": self.logits_pi,
      "probs": self.probs_pi
    }

    # We add entropy to the loss to encourage exploration
    self.entropy_pi = -tf.reduce_sum(self.probs_pi * tf.log(self.probs_pi), 1, name="entropy")

    # Get the predictions for the chosen actions only
    gather_indices_pi = tf.range(batch_size) * tf.shape(self.probs_pi)[1] + self.actions
    self.picked_action_probs_pi = tf.gather(tf.reshape(self.probs_pi, [-1]), gather_indices_pi)

    self.losses_pi = - (tf.log(self.picked_action_probs_pi) * self.targets_pi + 0.01 * self.entropy_pi)
    self.loss_pi = tf.reduce_sum(self.losses_pi, name="loss_pi")

    ### Value
    self.logits_v = tf.contrib.layers.fully_connected(
      inputs=fc1,
      num_outputs=1,
      activation_fn=None)
    self.logits_v = tf.squeeze(self.logits_v, squeeze_dims=[1])

    self.predictions_v = {
      "logits": self.logits_v
    }

    self.losses_v = tf.squared_difference(self.logits_v, self.targets_v)
    self.loss_v = tf.reduce_sum(self.losses_v, name="loss_v")

    # Combine loss
    self.loss = self.loss_pi + 0.5 * self.loss_v
    self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
    self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
    self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
      global_step=tf.contrib.framework.get_global_step())