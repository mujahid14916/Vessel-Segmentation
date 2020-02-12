import tensorflow as tf


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, activation=None):
        super(MyDenseLayer, self).__init__()
        self.num_outputs = num_outputs
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                shape=[input_shape[-1],
                                        self.num_outputs])
        self.bias = self.add_weight("bias", shape=[self.num_outputs])

    def call(self, inputs):
        result = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation == 'relu':
            return tf.keras.activations.relu(result)
        elif self.activation == 'sigmoid':
            return tf.keras.activations.sigmoid(result)
        else:
            return result

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(250, 250, 3)),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.Flatten(),
    MyDenseLayer(10, activation='relu'),
    MyDenseLayer(1, activation='sigmoid')
])

model.summary()

