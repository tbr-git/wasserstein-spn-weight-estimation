import tensorflow as tf 
import ot
import numpy as np
import time

from .pn_ot import PNConnectorOT
from .pn_ot import StochasticPath
from .stochastic_language.stochastic_lang import StochasticLang
from .stochastic_language.stochastic_lang import lvs_cost_matrix

class OptimalTransportWeightOptimizer:

    OPTIMIZATION_TIME_BOUND = 10

    def __init__(self, pn_ot_conn: PNConnectorOT, lang_log: StochasticLang):
        self.pn_ot_conn = pn_ot_conn
        self.lang_log = lang_log

    def optimize_weights(self):
        # Need missing probability source
        need_missing_prob_source = abs(self.pn_ot_conn.stoch_lang.probabilities.sum() - 1) > 0.0001
        C = lvs_cost_matrix(self.pn_ot_conn.stoch_lang, self.lang_log)

        if need_missing_prob_source:
            C = np.concatenate((C, np.ones((1, len(self.lang_log)))), axis=0)


        # Create transitions weight tensor
        t_weights = self.pn_ot_conn.tIdActConn.get_transition_weights()
        self.trans_weights = tf.Variable(np.array(t_weights), dtype=tf.float32, trainable=True, constraint=lambda x: tf.clip_by_value(x, 0.001, 50))
        #self.trans_weights = tf.Variable(np.ones(len(self.pn_ot_conn.net.transitions)), dtype=tf.float32, trainable=True, constraint=lambda x: tf.clip_by_value(x, 0.001, 50))


        ### Assumptions ###
        # Model has more than one path 
        self.error_series = []
        opt = tf.keras.optimizers.Adam(0.001)
        time_start = time.time()
        time_last_iteration = time_start

        ieration = 0
        #while time_last_iteration - time_start < OptimalTransportWeightOptimizer.OPTIMIZATION_TIME_BOUND:
        for i in range(10):
            print(f"Step {ieration}")

            with tf.GradientTape() as t:
                l_tf_variant_prob = []
                for i in range(len(self.pn_ot_conn.stoch_lang.variants)):
                    l_tf_path_prob = []
                    for p_index in self.pn_ot_conn.lang_2_path[i]:
                        path = self.pn_ot_conn.paths[p_index]
                        l_tf_fire_path = []

                        for (t_ind, enabled_ind) in zip(path.transition_ids, path.curr_enabled_transitions):

                            if len(enabled_ind) > 1:
                                tf_fire_prob = tf.divide(self.trans_weights[t_ind], 
                                                         tf.reduce_sum(tf.gather(self.trans_weights, enabled_ind)))
                                l_tf_fire_path.append(tf_fire_prob)

                        if len(l_tf_fire_path) > 1:
                            tf_path_prob = tf.reduce_prod(tf.stack(l_tf_fire_path))
                        else:
                            tf_path_prob = l_tf_fire_path[0]
                        l_tf_path_prob.append(tf_path_prob)

                    if len(l_tf_path_prob) > 1:
                        tf_variant_prob = tf.reduce_sum(tf.stack(l_tf_path_prob))
                    else:
                        tf_variant_prob = l_tf_path_prob[0]
                    l_tf_variant_prob.append(tf_variant_prob)

                if need_missing_prob_source:
                    l_tf_variant_prob.append(1 - tf.reduce_sum(tf.stack(l_tf_variant_prob)))

                self.tf_variant_prob = tf.stack(l_tf_variant_prob)
                loss = setup_loss_function(self.lang_log.probabilities, C)

                self.emd = loss(self.tf_variant_prob)

            # Optimization
            trainable_variables = [self.trans_weights]
            gradients = t.gradient(self.emd, trainable_variables)
            opt.apply_gradients(zip(gradients, trainable_variables))
            self.error_series.append(self.emd.numpy())

            ieration += 1
            time_last_iteration = time.time()

        return self.error_series

        #self.reg_loss = tf.add(self.emd, TODO)


def setup_loss_function(b, M):
    @tf.custom_gradient
    def emd_variant_likelihood_loss(a):
        (gamma, log) = ot.emd(a.numpy(), b, M, log=True)
        def grad(dy):
            return dy * log['u']
        return log['cost'], grad

    return emd_variant_likelihood_loss
