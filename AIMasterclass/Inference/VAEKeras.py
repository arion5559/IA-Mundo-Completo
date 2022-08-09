# Construcción del modelo VAE

import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

# Ajustando la dimension de entrada (tamaño de los frames)
INPUT_DIM = (64, 64, 3)

# Ajustando la dimensión de los vectores latentes
Z_DIM = 32

# Ajustando el tamaño de la capa densa
DENSE_SIZE = 1024

# Ajustando el número de epochs y el batch size
EPOCHS = 1
BATCH_SIZE = 32


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Z_DIM), mean=0, stddev=1.)
    return z_mean + K.exp(z_log_var/2) * epsilon


class ConvVAEKeras(object):
    # Inicialización de parámetros,
    # z_size: dimensión del vector latente
    # batch size: cuántos le paso antes de la back propagation (tamaño del bloque)
    # learning_rate: cómo de rápido aprende la red neuronal, cómo de rápido puede actualizar los pesos
    # kl_tolerance: suma de 2 pérdidas, error cuadrático medio y kl_loss
    # is_training: si está entrenando o en testing
    # reuse: reutilizar o no el ámbito de visibilidad de dónde viven las variables
    # gpu_mode: si usará la gpu o la cpu
    def __init__(self, input_dim=INPUT_DIM, z_dim=Z_DIM, dense_size=DENSE_SIZE, batch_size=BATCH_SIZE,
                 learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False, gpu_mode=False):
        self.models = self._build(input_dim, z_dim, dense_size)
        self.model = self.models[0]
        self.encoder = self.models[1]
        self.decoder = self.models[2]
        self.input_dim = input_dim
        self.z_dim = z_dim

    # Creación de un método para la arquitectura del modelo VAE
    def _build(self, input_dim, z_dim, dense_size):
        # Creando el modelo y el input del encoder
        vae_x = Input(shape=input_dim)
        # Creando las capas convolucionales del Encoder
        vae_c1 = Conv2D(filters=32, kernel_size=4, strides=2, activation=tf.nn.relu)(vae_x)
        vae_c2 = Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)(vae_c1)
        vae_c3 = Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)(vae_c2)
        vae_c4 = Conv2D(filters=128, kernel_size=4, strides=2, activation=tf.nn.relu)(vae_c3)
        # Aplanando la última capa convolucional
        vae_z_in = Flatten()(vae_c4)
        # Usando 2 archivos separados para calcular z_mean y z_log
        vae_z_mean = Dense(z_dim)(vae_z_in)
        vae_z_log_var = Dense(z_dim)(vae_z_in)
        # Usando la clase Lambda de Keras sobre la función que creamos arriba
        vae_z = Lambda(sampling)([vae_z_mean, vae_z_log_var])
        # Obteniendo los input del decoder
        vae_z_input = Input(shape=(z_dim,))
        # Instanciando estas capas de manera separada para poder volver a usarlas luego
        vae_dense = Dense(1024)
        vae_dense_model = vae_dense(vae_z)
        # Haciendo reshape a la capa para 4 dimensiones
        vae_z_out = Reshape((1, 1, dense_size))
        # Obteniendo el output de esta última capa
        vae_z_out_model = vae_z_out(vae_dense_model)
        # Haciendo el Decoder
        # Definiendo la capa convolucional
        vae_d1 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation=tf.nn.relu)
        # Creando la capa del decoder
        vae_d1_model = vae_d1(vae_z_out_model)
        vae_d2 = Conv2DTranspose(filters=64, kernel_size=5, strides=2, activation=tf.nn.relu)
        vae_d2_model = vae_d2(vae_d1_model)
        vae_d3 = Conv2DTranspose(filters=32, kernel_size=6, strides=2, activation=tf.nn.relu)
        vae_d3_model = vae_d3(vae_d2_model)
        vae_d4 = Conv2DTranspose(filters=3, kernel_size=6, strides=2, activation=tf.nn.relu)
        vae_d4_model = vae_d4(vae_d3_model)
        # Obteniendo el vector latente del decoder
        vae_dense_decoder = vae_dense(vae_z_input)
        vae_z_out_decoder = vae_z_out(vae_dense_decoder)
        vae_d1_decoder = vae_d1(vae_z_out_decoder)
        vae_d2_decoder = vae_d1(vae_d1_decoder)
        vae_d3_decoder = vae_d1(vae_d2_decoder)
        vae_d4_decoder = vae_d1(vae_d3_decoder)
        # Definiendo el modelo VAE compuesto del encoder y decoder
        vae = Model(vae_x, vae_d4_model)
        vae_encoder = Model(vae_x, vae_z)
        vae_decoder = Model(vae_z_input, vae_d4_decoder)

        # Implementando las operaciones de entrenamiento
        # Definiendo el MSE loss
        def vae_r_loss(y_true, y_pred):
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
            return 10 * K.mean(K.square(y_true_flat - y_pred_flat), axis=-1)

        # Definiendo el KL divergence loss
        def vae_kl_loss(y_true, y_pred):
            return -0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis=-1)

        # Definiendo la pérdida total del VAE, resumiendo el MSE y KL losses
        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)

    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, data, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE):
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]
        self.model.fit(data, data, shuffle=True, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split, callbacks=callbacks_list)
        self.model.save_weights('vae/weights.h5')

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    # Generando data para la MDN-RNN
    def generate_rnn_data(self, obs_data, action_data):
        rnn_input = []
        rnn_output = []
        for i, j in zip(obs_data, action_data):
            rnn_z_input = self.encoder.predict(np.array(i))
            conc = [np.concatenate([x, y]) for x, y in zip(rnn_z_input, j.reshape(-1, 1))]
            rnn_input.append(conc[:-1])
            rnn_output.append(np.array(rnn_z_input[1:]))
        rnn_input = np.array(rnn_input)
        rnn_output = np.array(rnn_output)
        print(f"Rnn inputs size: {rnn_input.shape}, Rnn outputs size: {rnn_output.shape}")
        return rnn_input, rnn_output
