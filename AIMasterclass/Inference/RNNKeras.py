import math
import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

Z_DIM = 32
ACTION_DIM = 3
HIDDEN_UNITS = 256
GAUSSIAN_MIXTURES = 5
BATCH_SIZE = 32
EPOCHS = 20


# Obteniendo los coeficientes de la mezcla gausiana
def get_mixture_coef(y_pred):
    d = GAUSSIAN_MIXTURES * Z_DIM
    rollout_length = K.shape(y_pred)[1]
    pi = y_pred[:, :, :, :d]
    mu = y_pred[:, :, d:(2*d)]
    log_sigma = y_pred[:, :, (2*d):(3*d)]
    pi = K.reshape(pi, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    mu = K.reshape(mu, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    log_sigma = K.reshape(log_sigma, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    pi = K.exp(pi) / K.sum(K.exp(pi), axis=2, keepdims=True)
    sigma = K.exp(log_sigma)
    return pi, mu, sigma


# Normalizando los valores objetivo
def tf_normal(y_true, mu, sigma, pi):
    rollout_length = K.shape(y_true)[1]
    y_true = K.tile(y_true, (1, 1, GAUSSIAN_MIXTURES))
    y_true = K.reshape(y_true, [-1, rollout_length, GAUSSIAN_MIXTURES, Z_DIM])
    oneDivSqrtTwoPi = 1/math.sqrt(2*math.pi)
    result = y_true - mu
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1 / (sigma + 1e-8))*oneDivSqrtTwoPi
    result = result * pi
    result = K.sum(result, axis=2)
    return result


class MDNRNNKeras(object):
    # Inicializando parámetros y variables para la MDNRNN
    def __init__(self):
        self.models = self._build()
        self.model = self.models[0]
        self.forward = self.models[1]
        self.z_dim = Z_DIM
        self.hidden_units = HIDDEN_UNITS
        self.gaussian_mixtures = GAUSSIAN_MIXTURES

    def build(self):
        # Definiendo los Inputs de la RNN (Vector latente espacial y espacio de acción
        rnn_x = Input(shape=(None, Z_DIM + ACTION_DIM))
        # Definiendo la capa LSTM que devuelve los pesos salientes y los estados de las celdas
        lstm = LSTM(HIDDEN_UNITS, return_sequences=True, return_state=True)
        # Obteniendo los resultados reales de la LSTM
        lstm_output, _, _ = lstm(rnn_x)
        # Obteniendo los resultados de la mezcla gausiana
        mdn = Dense(GAUSSIAN_MIXTURES * (3*Z_DIM))(lstm_output)
        # Obteniendo el modelo de entrenamiento
        rnn = Model(rnn_x, mdn)
        # Obteniendo el estado oculto y los input de celdas ocultas
        state_input_h = Input(shape=(HIDDEN_UNITS,))
        state_input_c = Input(shape=(HIDDEN_UNITS,))
        # Agrupándolos
        state_inputs = [state_input_h, state_input_c]
        # Obteniendo el nuevo estado de los outputs y el nuevo estado de celdas de la LSTM
        _, state_h, state_c = lstm(rnn_x, initial_state=state_inputs)
        # Definiendo la propagación hacia delante solo por inferencia
        forward = Model([rnn_x] + state_inputs, [state_h, state_c])
        # Implementando las operaciones de entrenamiento

        def rnn_r_loss(y_true, y_pred):
            # Definiendo la el logaritmo de la pérdida negativa sobre todas las mezclas gausianas
            pi, mu, sigma = get_mixture_coef(y_pred)
            result = tf_normal(y_true, mu, sigma, pi)
            result = -K.log(result + 1e-8)
            result = K.mean(result, axis=(1, 2))
            return result

        # Definiendo la pérdida de divergencia KL, la misma que en el VAE, solo outputs sobrenormalizados
        def rnn_kl_loss(y_true, y_pred):
            pi, mu, sigma = get_mixture_coef(y_pred)
            kl_loss = -0.5 * K.mean(1 + K.log(K.square(sigma)) - K.square(mu) - K.square(sigma), axis=[1, 2, 3])
            return kl_loss

        # Definiendo la pérdida de la RNN
        def rnn_loss(y_true, y_pred):
            return rnn_r_loss(y_true, y_pred)

        # Compilando el modelo RNN con la pérdida RNN y el optimizador RMSProp
        rnn.compile(loss=rnn_loss, optimizer="rmsprop", metrics=[rnn_r_loss, rnn_kl_loss])
        return (rnn, forward)

    # Cargando los pesos del modelo
    def set_weights(self, filepath):
        self.model.load_weights(filepath)

    # Creando las early stopping callbacks para prevenir el overfitting
    def train(self, rnn_input, rnn_output, validation_split=0.2):
        earlystop = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=5, verbose=1, mode="auto")
        callbacks_list = [earlystop]
        # Adecuando el modelo a los inputs de la RNN y objetivos
        self.model.fit(rnn_input, rnn_output,
                       shuffle=True,
                       epochs=EPOCHS,
                       batch_size=BATCH_SIZE,
                       validation_split=validation_split,
                       callbacks=callbacks_list)
        # Guardando el modelo después del entrenamiento
        self.model.save_weights("rnn/weights.h5")

    # Separando la función utilizada para guardar el modelo (útil por si se reentrena)
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
