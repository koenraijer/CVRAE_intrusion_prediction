import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, Concatenate, Flatten, RNN
from tensorflow.keras import backend as K
from tensorflow import shape
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow_addons.rnn import PeepholeLSTMCell
from tensorflow.keras.initializers import GlorotUniform
import os
import joblib

class LSTM_VAE(Model):
    def __init__(self, lstm_input_shape, int_dim=36, latent_dim=18, condition_dim=1, batch_size=32, learning_rate=0.001, reconstruction_wt = 2, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.lstm_input_shape = lstm_input_shape
        self.condition_dim = condition_dim
        self.int_dim = int_dim
        self.latent_dim = latent_dim
        self.num_features = lstm_input_shape[2]
        self.time_steps = lstm_input_shape[1]
        self.condition_dim = condition_dim
        self.condition_input = Input(shape=(condition_dim,)) # Shape of the condition input
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reconstruction_wt = reconstruction_wt
        self.seed = seed
        self.layer_seed = 0 
        if self.seed is not None:
            np.random.seed(self.seed)
            tf.random.set_seed(self.seed)
            
        # Everything needed for the encoder and decoder should be defined before calling the encoder and decoder functions
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    def build_encoder(self):
        """
        Builds the encoder part of the VAE.
        """

        def Sampling(args):
            """
            Sampling function for the VAE.
            Parameters: args (tuple): Tuple containing the mean and log variance of the latent space.
            Returns: Sampled latent space vector.
            """
            z_mean, z_log_var = args
            batch_size = shape(z_mean)[0] # Number of samples in the batch
            latent_dim = shape(z_mean)[1] # Dimensionality of the latent space
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0, stddev=1) # epsilon = irreducible error. Generates random noise to add to our reparameterisation trick value
            # Reparameterisation trick
            # - `K.exp()` takes the exponential of our log variance to obtain the variance. 
            # - `z_log_var / 2` is equivalent to taking the square root of the variance (standard deviation)
            # - `* epsilon` to get a random value from a normal distribution with mean 0 and standard deviation 1
            return z_mean + K.exp(z_log_var / 2) * epsilon

        # --------------------- ENCODER ---------------------
        # Creating the input layer
        x = Input(shape=(self.time_steps, self.num_features))  # Keras adds None to the shape for the batch size: (None, time_steps, number_of_features)
        encoder_input = Concatenate()([x, RepeatVector(self.time_steps)(self.condition_input)]) # Repeat condition for each timsestep. 

        # Creating the LSTM layers
        encoder_LSTM_intermediate = RNN(PeepholeLSTMCell(self.int_dim, kernel_initializer=GlorotUniform(seed=self.seed + self.layer_seed)), return_sequences=True, name="encoder_LSTM_intermediate")(encoder_input)
        self.layer_seed += 1
        encoder_LSTM_latent = RNN(PeepholeLSTMCell(self.latent_dim, kernel_initializer=GlorotUniform(seed=self.seed + self.layer_seed)), return_sequences=False, name="encoder_LSTM_latent")(encoder_LSTM_intermediate)
        self.layer_seed += 1

        # NOTE: GlorotUniform is equivalent to Xavier weights initialisation (recommended with tanh activation functions, the default in LSTMs).

        # These layers' outputs will be trained to represent the mean and log variance of the latent space
        z_mean = Dense(self.latent_dim, name="z_mean")(encoder_LSTM_latent) # Mean(s) of the latent space
        z_log_var = Dense(self.latent_dim, name="z_log_var")(encoder_LSTM_latent) # Log variance(s) of the latent space. Log is used to ensure so its exponent (which we'll calculate later) is always positive.

        # A Lambda layer is used to sample from the latent space by passing the mean and log variance to the vae_sampling function
        z = Lambda(Sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        encoder = Model(inputs=[x, self.condition_input], outputs=[z_mean, z_log_var, z], name="encoder") # z_mean and z_log_sigma are returned for loss calculation, z_encoder_output is the output of the encoder and will be used as input to the decoder

        # SIDENOTE: Keras is able to trace back the computation graph from the output of the encoder to the input, so it will infer the structure of the encoder from its output and input layers.

        return encoder

    def build_decoder(self):
        """
        Builds the decoder part of the VAE.
        """
        decoder_input = Input(shape=(self.latent_dim,)) # Input to the decoder is the latent space vector z
        condition_input_repeated = Flatten()(RepeatVector(self.latent_dim)(self.condition_input))
        decoder_input_concat = Concatenate()([decoder_input, condition_input_repeated])
        decoder_repeated = RepeatVector(self.time_steps)(decoder_input_concat) # Repeats the latent space vector (z) for the number of time steps, to match the input shape of the LSTM.
        decoder_LSTM_intermediate = RNN(PeepholeLSTMCell(self.int_dim, kernel_initializer=GlorotUniform(seed=self.seed + self.layer_seed)), return_sequences=True)(decoder_repeated) # Transforms (batch_size, time_steps, latent_dim) to (batch_size, time_steps, int_dim). 
        self.layer_seed += 1
        decoder_LSTM = RNN(PeepholeLSTMCell(self.num_features, kernel_initializer=GlorotUniform(seed=self.seed + self.layer_seed)), return_sequences=True)(decoder_LSTM_intermediate)
        decoder_output = TimeDistributed(Dense(self.num_features))(decoder_LSTM) # Contains Dense layer at the end to be able to produce high absolute values, since LSTM activations are tanh.
        decoder = Model(inputs=[decoder_input, self.condition_input], outputs=decoder_output, name="decoder") 
        
        return decoder

    @property
    def metrics(self):
            """
            Returns a list of metrics used for tracking during training.
            Returns:list: A list of metrics including total loss, reconstruction loss, and KL divergence loss.
            """
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]
        
    def call(self, inputs):
        """
        Performs the forward pass of the model.
        Args: inputs: The input data.
        Returns: The reconstructed output.
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder([z, inputs[1]])  # Added the condition input
        return reconstruction

    # def call(self, inputs):
    #     """
    #     Performs the forward pass of the model.
    #     Args: inputs: The input data.
    #     Returns: The reconstructed output.
    #     """
    #     z_mean, z_log_var, z = self.encoder(inputs)
    #     reconstruction = self.decoder(z)
    #     return reconstruction

    def calculate_losses(self, x, condition):
        """
        Calculates the total loss, reconstruction loss, and KL divergence loss for a given input and condition.

        Parameters:
            x (tensor): The input tensor.
            condition (tensor): The condition tensor.

        Returns:
            total_loss (tensor): The total loss.
            reconstruction_loss (tensor): The reconstruction loss.
            kl_loss (tensor): The KL divergence loss.
        """
        z_mean, z_log_var, z = self.encoder([x, condition])
        reconstruction = self.decoder([z, condition])
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mse(x, reconstruction),
                axis=(0, 1),
            )
        )
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        reconstruction_loss = self.reconstruction_wt * reconstruction_loss
        total_loss = reconstruction_loss + kl_loss
        return total_loss, reconstruction_loss, kl_loss

    @tf.function
    def train_step(self, data):
        """
        Performs a single training step on the given data.

        Args:
            data: A tuple containing the input data `x` and the condition `condition`.

        Returns:
            A dictionary containing the following metrics:
            - "loss": The total loss value.
            - "reconstruction_loss": The reconstruction loss value.
            - "kl_loss": The KL divergence loss value.
        """
        
        x, condition = data
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.calculate_losses(x, condition)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        """
        Perform a single testing step of the model.

        Args:
            data (tuple): A tuple containing the input data `x` and the condition `condition`.

        Returns:
            dict: A dictionary containing the calculated losses for the testing step, including the total loss, reconstruction loss, and KL loss.
        """
        x, condition = data
        
        total_loss, reconstruction_loss, kl_loss = self.calculate_losses(x, condition)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def save_weights(self, filepath):
        model_dir, filename = os.path.split(filepath)

        encoder_wts = self.encoder.get_weights()
        decoder_wts = self.decoder.get_weights()
        joblib.dump(encoder_wts, os.path.join(model_dir, f"{filename}_encoder_wts.h5"))
        joblib.dump(decoder_wts, os.path.join(model_dir, f"{filename}_decoder_wts.h5"))

    def load_weights(self, filepath):
        model_dir, filename = os.path.split(filepath)
        encoder_wts = joblib.load(os.path.join(model_dir, f"{filename}_encoder_wts.h5"))
        decoder_wts = joblib.load(os.path.join(model_dir, f"{filename}_decoder_wts.h5"))
        self.encoder.set_weights(encoder_wts)
        self.decoder.set_weights(decoder_wts)

    def save(self, filepath, overwrite=True, options=None):
        model_dir, filename = os.path.split(filepath)
        
        params_file = os.path.join(model_dir, f"{filename}_parameters.pkl")

        if not overwrite and os.path.exists(weights_file) and os.path.exists(params_file):
            return

        self.save_weights(filepath)
        
        dict_params = {
            # Optimised params
            "batch_size": self.batch_size,
            "int_dim": self.int_dim,
            "latent_dim": self.latent_dim,
            "learning_rate": self.learning_rate,
            "reconstruction_wt": self.reconstruction_wt, 
            # "optimizer" : self.optimizer.get_config()['name'],  # Save only the name of the optimizer
            # Structural params
            "condition_dim" : self.condition_dim,
            "input_shape" : self.lstm_input_shape,
            "seed" : self.seed
        }
        
        joblib.dump(dict_params, params_file)

    @classmethod # https://github.com/abudesai/timeVAE/blob/cadc1098ea48896faaaf813d96e28575747dddb5/vae_dense_model.py#L57
    def load(cls, filepath):
        model_dir, filename = os.path.split(filepath)
        params_file = os.path.join(model_dir, f"{filename}_parameters.pkl")
        params = joblib.load(params_file)
        print(params)
    
        # Deconstruct the params dictionary
        batch_size = params['batch_size']
        int_dim = params['int_dim']
        latent_dim = params['latent_dim']
        learning_rate = params['learning_rate']
        reconstruction_wt = params['reconstruction_wt']
        condition_dim = params['condition_dim']
        input_shape = params['input_shape']
        seed = params['seed']
    
        # Create the model
        model = LSTM_VAE(batch_size=batch_size, int_dim=int_dim, latent_dim=latent_dim, learning_rate=learning_rate, 
                         reconstruction_wt=reconstruction_wt, condition_dim=condition_dim, lstm_input_shape=input_shape, seed=seed)
        
        model.load_weights(filepath)
    
        # Set the optimizer
        optimizer = Adam(learning_rate=learning_rate)  # We're setting optimizer=0 (Adam) by hand
        model.compile(optimizer=optimizer)
        
        return model
        
    def generate_samples(self, n, condition):
        '''
            Generate random samples from the LSTM VAE.

            n : int : The number of samples to generate.
            condition : numpy array : The condition to generate the samples for.

            Returns
            -------
            A numpy array of shape (n, time_steps, number_of_features) containing the generated samples.
        '''

        # Sample from the standard normal distribution
        z_samples = np.random.normal(size=(n, self.latent_dim))

        # Decode the samples
        gen = self.decoder.predict([z_samples, np.repeat(condition, n, axis=0)])

        return gen
        
class PrintModelPerEpoch(tf.keras.callbacks.Callback):
    def __init__(self, print_weights=False):
        super().__init__()
        self.print_weights = print_weights

    def on_epoch_begin(self, epoch, logs=None):
        print("##########################################")
        for model in [self.model.encoder, self.model.decoder]:
            print(f"Model: {model.name}")
            for layer in model.layers:
                print(f"{layer.name}, input shape: {layer.input_shape}")
                if self.print_weights and layer.weights:
                    for weight in layer.weights:
                        weight_values = weight.numpy()
                        print(f"{layer.name}, weights: {weight.name}, mean: {weight_values.mean()}, variance: {weight_values.var()}")
            if model == self.model.encoder: print("------------------------------------------")
            else: print("##########################################")