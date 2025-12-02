import tensorflow as tf

'''
Class for RTI neural models.

    Parameters
    ----------
    num_inputs : int
        Number of input, for RTI will be most probably 3 times the number of source images
    encoder_parameters : list[int, ...]
        Array containing, as a sequence, the number of parameters for each encoder linear layer
    decoder_parameters : list[int, ...]
        Array containing, as a sequence, the number of parameters for each decoder linear layer
    comp_coeff: int
        Number of computed coefficients, per pixel, in the latent space
    activation_function: string
        Activation function for each linear layer (it adds non-linearities)
    light_dimension: int
        Number of light dimensions for RTI: 2 -> (x,y) | 3 -> (x,y,z)
    position_dimension: int
        Number of pixel-position dimensions: 0 -> no position | 2 -> (u,v)
    num_outputs: int
        Number of outputs from the decoder, for RTI is commonly 3 (RGB channels)

    Returns
    -------
'''

class relighting_model(tf.keras.Model):
    def __init__(self, num_inputs, encoder_parameters, decoder_parameters, comp_coeff = 9, activation_function = 'elu', light_dimension = 2, position_dimension = 0, num_outputs = 3):
        super().__init__()
        self.num_inputs = num_inputs                        # number of input, for RTI will be most probably 3 times the number of source images
        self.encoder_parameters = encoder_parameters        # array containing, as a sequence, the number of parameters for each encoder linear layer
        self.decoder_parameters = decoder_parameters        # array containing, as a sequence, the number of parameters for each decoder linear layer
        self.comp_coeff = comp_coeff                        # number of computed coefficients, per pixel, in the latent space
        self.activation_function = activation_function      # activation function for each linear layer (it adds non-linearities)
        self.light_dimension = light_dimension              # number of light dimensions for RTI: 2 -> (x,y) | 3 -> (x,y,z)
        self.position_dimension = position_dimension        # number of pixel-position dimensions: 0 -> no position | 2 -> (u,v)
        self.num_outputs = num_outputs                      # number of outputs from the decoder, for RTI is commonly 3 (RGB channels)
        self.encoder, self.decoder = self.standard_model()  # create two separate variables for encoder and decoder

    def call(self, inputs):

        # sample: concatenation of the pixel in all light configurations
        # light: light direction for the current ground truth pixel
        # features: latent space computed by the encoder
        # output: final pixel computed by the decoder, usually RGB

        sample, light = inputs
        #features = self.encoder(tf.keras.layers.concatenate([sample,light], axis=-1))
        features = self.encoder(sample)
        output = self.decoder(tf.keras.layers.concatenate([features, light], axis=-1))
        return output

    def standard_model(self):

        # layers1: list of linear layers for the encoder
        # layers2: list of linear layers for the decoder

        layers1 = [tf.keras.Input(shape=(self.num_inputs,))]
        for size in self.encoder_parameters:
            layers1.append(tf.keras.layers.Dense(units=size, activation=self.activation_function))
        layers1.append(tf.keras.layers.Dense(units=self.comp_coeff))

        layers2 = [tf.keras.Input(shape=(self.comp_coeff+self.light_dimension+self.position_dimension,))]
        for size in self.decoder_parameters:
            layers2.append(tf.keras.layers.Dense(units=size, activation=self.activation_function))
        layers2.append(tf.keras.layers.Dense(units=self.num_outputs))

        # encoder and decoder are defined as two separate "Sequential" keras model
        encoder = tf.keras.Sequential(layers=layers1, name='encoder')
        decoder = tf.keras.Sequential(layers=layers2, name='decoder')

        # not sure if this compilation is required
        encoder.compile(optimizer='adam', loss='mean_squared_error')
        decoder.compile(optimizer='adam', loss='mean_squared_error')

        return encoder, decoder