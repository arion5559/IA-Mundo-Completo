import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import Graph
from tensorflow import nn

ACTIONS = 3


# Construyendo el modelo MDN-RNN en una clase
class MDNRNN(object):
    def __init__(self, hps, reuse=False, gpu_mode=False):
        self.hps = hps
        self.reuse = reuse
        with tf.variable_scope('mdn_rnn', reuse=self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.compat.v1.logging.info('Model using cpu')
                    self.g = Graph()
                    if self.g.as_default():
                        self.build_model(hps)
            else:
                tf.compat.v1.logging.info('Model using gpu')
                gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4, allow_growth=True)
                config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
                sess = tf.compat.v1.Session(config=config)
                tf.compat.v1.keras.backend.set_session(sess)
                self.g = Graph()
                if self.g.as_default():
                    self.build_model(hps)
        self._init_session()

    # Creando un método que cree el modelo
    def build_model(self, hps):
        # Construcción de la RNN
        self.num_mixtures = hps.num_mixtures
        KMIX = self.num_mixtures
        INWIDTH = hps.input_seq_width
        OUTWIDTH = hps.input_seq_width
        LENGTH = hps.max_seq_len
        if hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
        cell_fn = tfa.rnn.LayerNormLSTMCell()
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        use_layer_norm = False if self.hps.use_layer_norm == 0 else True
        if use_recurrent_dropout:
            cell = cell_fn(hps.rnn_size, use_bias=use_layer_norm, dropout=self.hps.recurrent_dropout_prob)
        else:
            cell = cell_fn(hps.rnn_size, use_bias=use_layer_norm)
        if use_input_dropout:
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
        self.cell = cell
        self.seq_length = LENGTH
        self.input_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.seq_length, INWIDTH])
        self.output_x = tf.compat.v1.placeholder(dtype=tf.float32,
                                                 shape=[self.hps.batch_size, self.seq_length, OUTWIDTH])
        actual_input_x = self.input_x
        self.initial_state = cell.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32)
        N_OUT = OUTWIDTH * KMIX * ACTIONS
        with tf.compat.v1.variable_scope("RNN"):
            output_w = tf.compat.v1.get_variable("output_w", shape=[self.hps.rnn_size, N_OUT])
            output_b = tf.compat.v1.get_variable("output_b", shape=[N_OUT])
        output, last_state = tf.compat.v1.nn.dynamic_rnn(cell=cell,
                                                         inputs=self.input_x,
                                                         parallel_iterations=None,
                                                         swap_memory=True,
                                                         scope="RNN"
                                                         )
        # Construcción de la MDN
        output = tf.reshape(output, shape=[-1, hps.rnn_size])
        output = tf.compat.v1.nn.xw_plus_b(x=output, weights=output_w, biases=output_b)
        output = tf.reshape(output, shape=[-1, KMIX * ACTIONS])
        self.final_state = last_state

        def get_mdn_coef(output):
            logmix, mean, logstd = tf.split(value=output, num_or_size_splits=3, axis=1)
            logmix = logmix - tf.math.reduce_logsumexp(input_tensor=logmix, axis=1, keepdims=True)
            return logmix, mean, logstd
        out_logmix, out_mean, out_logstd = get_mdn_coef(output)
        self.out_logmix = out_logmix
        self.out_mean = out_mean
        self.out_logstd = out_logstd
        # Implementar la operación de dentrenamiento
        
