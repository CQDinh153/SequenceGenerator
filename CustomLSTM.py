from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import BasicLSTMCell, LSTMStateTuple, tanh
import tensorflow as tf


class MyLSTM(BasicLSTMCell):
    def __init__(self, num_units, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tanh):
        BasicLSTMCell.__init__(self, num_units, forget_bias, input_size, state_is_tuple, activation)

        self.W = None
        self.b = None

    def link_weights(self, other):
        self.W = tf.identity(other.W)
        self.b = tf.identity(other.b)

    def set_weights(self, sess, W, b):
        sess.run(tf.assign(self.W, W))
        sess.run(tf.assign(self.b, b))

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""

        with vs.variable_scope(scope or "basic_lstm_cell"):
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)

            """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

              Args:
                args: a 2D Tensor or a list of 2D, batch x n, Tensors.
                output_size: int, second dimension of W[i].
                bias: boolean, whether to add a bias term or not.
                bias_start: starting value to initialize the bias; 0 by default.
                scope: (optional) Variable scope to create parameters in.

              Returns:
                A 2D Tensor with shape [batch x output_size] equal to
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

              Raises:
                ValueError: if some of the arguments has unspecified or wrong shape.
              """

            shapes = [inputs.get_shape(), h.get_shape()]
            total_arg_size = shapes[0][1].value + shapes[1][1].value

            dtype = inputs.dtype

            # Now the computation.
            if self.W is None:
                weights = vs.get_variable("weights", [total_arg_size, 4 * self._num_units], dtype=dtype)
                self.W = weights
            else:
                weights = self.W
            res = tf.matmul(array_ops.concat([inputs, h], 1), weights)

            tf.get_variable_scope().set_partitioner(None)

            if self.b is None:
                biases = tf.get_variable("biases", [4 * self._num_units], initializer=tf.constant_initializer(0, dtype=dtype), dtype=dtype)
                self.b = biases
            else:
                biases = self.b
            concat = tf.nn.bias_add(res, biases)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat([new_c, new_h], 1)
            return new_h, new_state

    def get_weights(self, sess):
        return sess.run(self.W)

    def get_biases(self, sess):
        return sess.run(self.b)