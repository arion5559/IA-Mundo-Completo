import numpy as np
import tensorflow as tf
from tensorflow import Graph


# Construyendo el modelo CNN-VAE en una clase

class ConvVAE(object):

    # Inicialización de parámetros,
    # z_size: dimensión del vector latente
    # batch size: cuántos le paso antes de la back propagation (tamaño del bloque)
    # learning_rate: cómo de rápido aprende la red neuronal, cómo de rápido puede actualizar los pesos
    # kl_tolerance: suma de 2 pérdidas, error cuadrático medio y kl_loss
    # is_training: si está entrenando o en testing
    # reuse: reutilizar o no el ámbito de visibilidad de dónde viven las variables
    # gpu_mode: si usará la gpu o la cpu
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False,
                 gpu_mode=False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse
        with tf.variable_scope('conv_vae', reuse=self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.compat.v1.logging.info('Model using cpu')
                    self._build_graph()
            else:
                tf.compat.v1.logging.info('Model using gpu')
                gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
                config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
                sess = tf.compat.v1.Session(config=config)
                tf.compat.v1.keras.backend.set_session(sess)
                self._build_graph()
        self._init_session()

    # Creando un método que cree el modelo
    def _build_graph(self):
        self.g = Graph()
        with self.g.as_default():
            self.x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64, 64, 3])
            # Construyendo el encoder
            h = tf.nn.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
            h = tf.nn.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
            h = tf.nn.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
            h = tf.nn.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
            h = tf.reshape(h, [-1, 2*2*256])
            # Building the "V" part of the VAE
            self.mu = tf.nn.dense(h, self.z_size, name="enc_fc_mu")
            self.logvar = tf.nn.dense(h, self.z_size, name="enc_fc_log_var")
            self.sigma = tf.exp(self.logvar / 2.0)
            self.epsilon = tf.random.normal([self.batch_size, self.z_size])
            self.z = self.mu + self.sigma * self.epsilon
            # Construyendo el decoder
            h = tf.nn.dense(self.z, 1024, name="dec_fc")
            h = tf.reshape(h, [-1, 1, 1, 1024])
            h = tf.nn.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
            h = tf.nn.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
            h = tf.nn.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
            self.y = tf.nn.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")
            # Implementando las operaciones de entrenamiento
            if self.is_training:
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.r_loss = tf.reduce_sum(tf.square(self.x - self.y), reduction_indices=[1, 2, 3])
                self.r_loss = tf.reduce_mean(self.r_loss)
                self.kl_loss = -0.5*tf.reduce_sum(1+self.logvar-tf.square(self.mu)-tf.exp(self.logvar),
                                                  reduction_indices=1)
                self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance*self.z_size)
                self.kl_loss = tf.reduce_mean(self.kl_loss)
                self.loss = self.r_loss + self.kl_loss
                self.lr = tf.Variable(self.learning_rate, trainable=False)
                self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
                grads = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step')
            self.init = tf.compat.v1.global_variables_initializer()
