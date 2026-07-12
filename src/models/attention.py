import tensorflow as tf
from keras.layers import Layer

class TemporalAttention(Layer):
    """
    Original Keras temporal attention layer.
    """
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(1,), initializer="zeros", trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        # x: (batch_size, time_steps, hidden_dim)
        # tf.matmul works over batched tensor
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        
        # Normalize over time_steps
        alpha = tf.nn.softmax(e, axis=1)
        
        # Weighted temporal representation
        context = x * alpha
        context = tf.reduce_sum(context, axis=1)
        
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
        
    def get_config(self):
        return super(TemporalAttention, self).get_config()
