import tensorflow as tf

class NetModel(tf.keras.Model):
    
    def __init__(self, num_states, hidden_units, num_actions):
        super(NetModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, activation='relu', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        x = self.input_layer(inputs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        output = self.output_layer(x)
        return output
