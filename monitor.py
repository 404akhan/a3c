import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf
import time

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

from gym.wrappers import Monitor
import gym

from lib.atari.state_processor import StateProcessor
from lib.atari import helpers as atari_helpers
from estimators import ValueEstimator, PolicyEstimator
from worker import make_copy_params_op


class Monitor(object):
  def __init__(self, summary_writer, saver):
  	self.saver = saver
  	self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))

  def save_once(self, sess):
    with sess.as_default(), sess.graph.as_default():
      self.saver.save(sess, self.checkpoint_path)

  def continuous_save(self, eval_every, sess, coord):
    """
    Continuously saves the policy every [eval_every] seconds.
    """
    try:
      while not coord.should_stop():
        self.save_once(sess)
        # Sleep until next evaluation cycle
        time.sleep(eval_every)
    except tf.errors.CancelledError:
      return