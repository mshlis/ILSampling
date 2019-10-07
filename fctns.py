import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.backend import batch_dot

def norm_derivative(dy, sparse_draws):
    """
    helper function for going through the derivative of unit normalization
    """
    sparse_draws = tf.cast(sparse_draws, tf.int32)
    dv = tf.gather_nd(dy, tf.stack([tf.range(tf.shape(sparse_draws)[0]), sparse_draws], -1))
    dv = tf.tile(tf.expand_dims(dv,-1), (1, tf.shape(dy)[-1]))
    return dy - dv

def reloss_draw(logits, const=1.):
    """
    ReLoss drawing
    """
    @tf.custom_gradient
    def _reloss_draw(logits):
        sparse_draws = tf.random.categorical(logits, num_samples=1)
        sparse_draws = tf.squeeze(sparse_draws, -1)
        draws = tf.one_hot(sparse_draws, depth=tf.shape(logits)[-1], dtype=tf.float32)
        def grad(dy):
            dy = norm_derivative(dy, sparse_draws)
            new = draws - const*dy
            new = new / tf.reduce_sum(new, axis=-1, keep_dims=True)
            return - new / (tf.nn.softmax(logits) + 1e-7)
        return draws, grad
    return _reloss_draw(logits)

def gumbel_softmax(logits, temp=1.):
    """
    Gumbel Softmax trick using tfp implementation
    """
    dist = tfp.distributions.RelaxedOneHotCategorical(temp, logits=logits)
    return dist.sample()
    