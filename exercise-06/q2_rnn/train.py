import tensorflow as tf
import logging


def run_training(sess, nbr_epochs, train_op, total_loss, it_train_init):
    """

    :param sess: TensorFlow session used for training
    :param nbr_epochs: Number of training epochs
    :param train_op: Training step operation
    :param total_loss: Batch loss
    :param it_train_init: Training dataset iterator initializer
    :return:
    """

    sess.run(tf.global_variables_initializer())
    # Run training epochs
    for e in range(nbr_epochs):
        sess.run(it_train_init)
        loss_acc = 0.0
        n_batch = 0
        try:
            while True:
                _, cur_loss = sess.run([train_op, total_loss])
                n_batch += 1
                loss_acc += cur_loss
                if n_batch % 50 == 0:
                    logging.info('Epoch %d: Train batch %d loss: %f' % (e, n_batch, cur_loss))
        except tf.errors.OutOfRangeError:
            logging.info('End of epoch %d: Avg loss %f' % (e, loss_acc / n_batch))


def get_loss_sin(outputs, y):
    """ Loss for the sin learning task. Mean squared error loss is used

    :param outputs: Model outputs
    :param y: Expected outputs
    :return: Mean squared error
    """

    #####Insert your code here for subtask 2c#####

    return loss


def get_train_op(loss, lr):
    """ Training operation is simply gradient descent with momentum.

    :param loss: Loss tensor
    :param lr: Learning rate
    :return:
    """

    #####Insert your code here for subtask 2d#####

    return train_op


def get_loss_memory(outputs, y):
    """ Get loss for the memory task.

    :param outputs: Model outputs
    :param y: Expected labels
    :return: Batch loss
    """
    #####Insert your code here for subtask 2e#####
    return loss
