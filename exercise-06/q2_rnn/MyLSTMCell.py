import tensorflow as tf
import collections
import RNNCell

MyLSTMState = collections.namedtuple('MyLSTMState', ['cell', 'hidden'])


class MyLSTMCell(RNNCell.RNNCell):
    """ Implementation of a basic LSTM cell.

    """

    def __init__(self, n_inp, n_lstm):
        """

        :param n_inp:  Dimension of the input
        :param n_lstm: Size of lstm cell
        """
        super(MyLSTMCell, self).__init__(n_inp, n_lstm)
        self.n_lstm = n_lstm

        with tf.variable_scope('LSTM_Cell'):
            #####Insert your code here for subtask 2a#####
            pass

    def __call__(self, state, x):
        """ Apply the cell to an given input i.e. unroll the RNN on the input and add it to the graph.

        :param state: Initial state
        :param x: Input (Batch Size, Time steps, Input dimension)
        :return: List of outputs of every step, tensor of the final state after computation
        """
        # In case one-hot encoding is used
        x = tf.cast(x, tf.float32)

        #####Insert your code here for subtask 2b#####
        return outputs, state

    def get_state_placeholder(self, name):
        """

        :param name: Name of the placeholder
        :return: Placeholder for the state (cell and hidden state)
        """
        ph_cell = tf.placeholder(dtype=tf.float32, shape=(None, self.n_lstm), name='%s_%s' % (name, 'cell'))
        ph_hidden = tf.placeholder(dtype=tf.float32, shape=(None, self.n_lstm), name='%s_%s' % (name, 'hidden'))
        return MyLSTMState(ph_cell, ph_hidden)

    def get_zero_state_for_inp(self, x):
        """ Return a initial zero state for the cell based on the expected batch size.

        :param x: Expected input (Used to determine the batch size)
        :return: Zero initial state for the expected batch size
        """
        zeros_dim = tf.stack([tf.shape(x)[0], self.n_lstm])
        return MyLSTMState(tf.zeros(zeros_dim), tf.zeros(zeros_dim))

    def create_state_feed_dict(self, pl_state, np_state):
        """ Return a dict which contains contains state assignments for tensorflows feed_dict

        :param pl_state: Data structure containing the state placeholders (cell and hidden state)
        :param np_state: Data for the state placeholders
        :return: Dict feeding the state data into the placeholders
        """
        return {pl_state.cell: np_state.cell, pl_state.hidden: np_state.hidden}
