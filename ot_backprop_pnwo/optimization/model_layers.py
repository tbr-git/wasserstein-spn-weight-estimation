import tensorflow as tf
import numpy as np
import numpy.typing as npt
import logging
from typing import Union

class Path2VariantProbabilityLayer(tf.keras.layers.Layer):

    MAX_TRANSITION_WEIGHT = 50

    MIN_INITIAL_TRANSITION_WEIGHT = 0.00001

    def __init__(self, 
                 w_transitions_init: Union[npt.NDArray[np.float_], None]=None,
                 nbr_transitions: Union[int, None]=None,
                 tw_clip_min: Union[int, None]=None, 
                 tw_clip_max=None,
                 **kwargs):
        super().__init__(trainable=True, **kwargs)

        self._adds_extra_losses = False
        # Transition weight variable
        self._w_transitions_init = w_transitions_init 
        self._nbr_transitions = nbr_transitions 
        self._tw_clip_min = tw_clip_min 
        self._tw_clip_max = tw_clip_max



    def build(self, input_shape):
        self.w_transitions = self._init_transition_weights(self._w_transitions_init, self._nbr_transitions, 
                                                           self._tw_clip_min, self._tw_clip_max)
        # Add as additional trainable variable
        #self.trainable_variables.extend([self.w_transitions])

    def _init_transition_weights(self,
                                 w_transitions_init: Union[npt.NDArray[np.float_], None]=None,
                                 nbr_transitions: Union[int, None]=None,
                                 tw_clip_min: Union[int, None]=None, tw_clip_max=None):

        # No array of values provided
        if w_transitions_init is None:
            if nbr_transitions is None:
                raise ValueError("Neither transition weights nor number of transitions is provided for initialization")
            else:
                w_transitions_initializer = self._get_tw_initializer()            
        else:
            nbr_transitions = len(w_transitions_init)
            w_transitions_initializer = self._adapted_hot_start_weight_initializer(w_transitions_init)

        # Create variables here because I latter access them
        # Easiest way to use the custom weights to initialize them
        # (Keras automaticall picks up all variables that are member of this class instance)
        if tw_clip_min is not None or tw_clip_max is not None:
            constraint = lambda x: tf.clip_by_value(x, 0 if tw_clip_min is None else tw_clip_min, Path2VariantProbabilityLayer.MAX_TRANSITION_WEIGHT if tw_clip_max is None else tw_clip_max)
        else:
            constraint = None
            # Clip transition weights (does not make a lot of sense if log_barrier term is applied, yet we would still do it)
            #w_transitions = tf.Variable(w_transitions_init, dtype=tf.float32, 
            #    trainable=True, 
            #    constraint=lambda x: tf.clip_by_value(x, 
            #                                0 if tw_clip_min is None else tw_clip_min, 
            #                                Path2VariantProbabilityLayer.MAX_TRANSITION_WEIGHT if tw_clip_max is None else tw_clip_max))
            #w_transitions = tf.Variable(w_transitions_init, dtype=tf.float32, trainable=True)
        w_transitions = self.add_weight(name="transition_weights", shape=[nbr_transitions], 
                                        initializer=w_transitions_initializer,
                                        dtype=tf.float32, constraint=constraint)

        return w_transitions
    
    def _get_tw_initializer(self):
        return tf.random_uniform_initializer(minval=0.25, maxval=1.75)

    def _adapted_hot_start_weight_initializer(self, w_transitions_init: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        # Ensure that transition weights are non-zero
        w_transitions_init = np.where(w_transitions_init > 0, w_transitions_init, Path2VariantProbabilityLayer.MIN_INITIAL_TRANSITION_WEIGHT)
        # Avoid huge weights
        w_transitions_init = w_transitions_init / np.max(w_transitions_init)
        return tf.constant_initializer(w_transitions_init)

    @property
    def transition_weights(self):
        """The transition_weights property."""
        return self.w_transitions.numpy()

    @property
    def adds_extra_losses(self):
        return self._adds_extra_losses


class Path2VariantProbabilityLayerBase(Path2VariantProbabilityLayer):

    def __init__(self, w_transitions_init: Union[npt.NDArray[np.float_], None],
                 nbr_transitions: Union[int, None]=None, 
                 tw_absolute: bool=False,
                 tw_log_barrier_mu: Union[float, None]=None,
                 tw_regularize_f: Union[float, None]=None, 
                 tw_clip_min: Union[float, None]=None, tw_clip_max: Union[float, None]=None, 
                 **kwargs):

        super().__init__(w_transitions_init=w_transitions_init, 
                         nbr_transitions=nbr_transitions, 
                         tw_clip_min=tw_clip_min, tw_clip_max=tw_clip_max, **kwargs)
        # Logger
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.debug('Adding path to variant layer - '\
                          f'absolute transition weights ({tw_absolute}); '\
                          f'transition weight clipping ({tw_clip_min}, {tw_clip_max}); '\
                          f'log barrier function ({tw_log_barrier_mu}); '\
                          f'weight regularization ({tw_regularize_f});')


        # Use absolute transition weights to avoid negative values
        self._tw_absolute = tw_absolute,
        # Log barrier
        self._tw_log_barrier_mu = tw_log_barrier_mu
        # Weight regularization
        self._tw_regularize_f = tw_regularize_f

        self._adds_extra_losses = self._tw_log_barrier_mu is not None or self._tw_regularize_f is not None


    def call(self, inputs):
        variant_2_paths = inputs[0]
        paths_nom = inputs[1]
        paths_denom = inputs[2]

        transition_weights = self.w_transitions
        # Does not work, weird shape error
        #transition_weights = tf.math.abs(self.w_transitions)

        ### Path Probabilities
        # For each variant, which paths result in this variant
        # CAN be an empty list

        # Denominator sum of transition weights for each path's factor
        paths_denom_weight_sum = tf.gather_nd(transition_weights, tf.expand_dims(paths_denom, -1))
        if self._tw_absolute:
            paths_denom_weight_sum = tf.math.abs(paths_denom_weight_sum)
        paths_denom_weight_sum = tf.reduce_sum(paths_denom_weight_sum, axis=-1)

        # Transition weights for each "step" in path
        paths_nom_weight = tf.gather_nd(transition_weights, 
                                        tf.expand_dims(paths_nom, -1))
        
        if self._tw_absolute:
            paths_nom_weight = tf.math.abs(paths_nom_weight)

        # Factor for each path step 
        paths_factors = tf.divide(paths_nom_weight, paths_denom_weight_sum)

        # Paths probabilities
        paths_probabilities = tf.math.reduce_prod(paths_factors, axis=-1)

        ### Variant Probabilities
        # Sum over associated paths
        variant_probabilities = tf.reduce_sum(
            tf.gather_nd(paths_probabilities, tf.expand_dims(variant_2_paths, -1)), 
            axis=-1)

        ### Losses
        # Log-barrier loss
        if self._tw_log_barrier_mu:
            if self._tw_absolute:
                self.add_loss(-1 * self._tw_log_barrier_mu * tf.reduce_sum(tf.math.log(tf.math.abs(transition_weights))))
            else:
                self.add_loss(-1 * self._tw_log_barrier_mu * tf.reduce_sum(tf.math.log(transition_weights)))
        
        # Weight regularization 
        if self._tw_regularize_f:
            self.add_loss(self._tw_regularize_f * tf.reduce_sum(tf.math.square(1 - transition_weights)))

        return variant_probabilities


    @property
    def transition_weights(self):
        """The transition_weights property."""
        if self._tw_absolute:
            return np.abs(self.w_transitions.numpy())
        else:
            return self.w_transitions.numpy()


class Path2VariantProbabilityLayerLogDomain(Path2VariantProbabilityLayer):

    def __init__(self, w_transitions_init: Union[npt.NDArray[np.float_], None],
                 nbr_transitions: Union[int, None]=None, 
                 log_domain: bool=False,
                 tw_absolute: bool=False,
                 tw_clip_min: Union[float, None]=None, tw_clip_max: Union[float, None]=None, 
                 **kwargs):
        self._logger = logging.getLogger(self.__class__.__name__)
        ##############################
        # Post Process configuration
        ##############################
        # To obtain a feasible EMD problem, we must keep the transition weights non-negative
        # This layer implements the following options:
        # 1. Optimization in log domain (optimize the transition weight logits)
        # 2. Use absolute transition weights
        # 3. Transition weights clipping
        # =>
        # Log-domain has highest priority. In log domain weights cannot become negative -> adapt other parameters accordingly
        # If absolute transition weights are used, we do not need clipping -> adapt clipping accordingly
        if log_domain:
            if tw_absolute:
                self._logger.warning("Setting absolut transition weights to False. Running the weight optimization in log domain, it is not adivisable to use absolute transition weights.")
                tw_absolute = False
            if tw_clip_min is not None or tw_clip_max is not None: 
                self._logger.warning("Disable transition weight clipping. Running the weight optimization in log domain, clipping is not needed to avoid infeasible weights.")
                tw_clip_min = None
                tw_clip_max = None
        elif tw_absolute:
            if tw_clip_min is not None or tw_clip_max is not None: 
                self._logger.warning("Disable transition weight clipping. Using absolute transiton weights, clipping is not needed to avoid infeasible weights.")
                tw_clip_min = None
                tw_clip_max = None
                

        self._use_log_domain_weights = log_domain
        self._tw_absolute = tw_absolute


        self._logger.debug('Adding log-powered path to variant layer - '\
                          f'log domain ({self._use_log_domain_weights}); '\
                          f'absolute transition weights ({self._tw_absolute}); '\
                          f'transition weight clipping ({tw_clip_min}, {tw_clip_max});')

        super().__init__(w_transitions_init=w_transitions_init, nbr_transitions=nbr_transitions, 
                         tw_clip_min=tw_clip_min, tw_clip_max=tw_clip_max, **kwargs)


    def call(self, inputs):
        variant_2_paths = inputs[0]
        paths_nom = inputs[1]
        paths_denom = inputs[2]
        transition_weights = self.w_transitions

        ### Gather nomination and denominator terms
        if self._use_log_domain_weights:
            # Each path's probability:
            # t_{i1} / (sum_{t is enabled} t) * t_{i2} / (sum{t is enabled} t) * ... 
            # Log domain with z_{i1} = log(t_{i1}):
            # exp(z_{i1} - log(sum_{t is enabled} (t)) + ...)
            # = exp(z_{i1} - log(sum_{t is enabled} (exp(z_t)) + ...) (log-sum-exp term!)

            paths_denom_log_weights = tf.gather_nd(transition_weights, tf.expand_dims(paths_denom, -1))
            # Log-sum-exp trick
            c = tf.math.reduce_max(paths_denom_log_weights, axis=-1, keepdims=True)
            c_squeezed = tf.squeeze(c, axis=-1)
            paths_denom_log_weights = paths_denom_log_weights - c
            paths_denom_log_weights = tf.math.exp(paths_denom_log_weights)
            paths_denom_log_weights = tf.reduce_sum(paths_denom_log_weights, axis=-1)
            paths_denom_log_weights = tf.math.log(paths_denom_log_weights)
            paths_denom_log_weight_sum = c_squeezed + paths_denom_log_weights
            # Denominator sum of transition weights for each path's factor
            # Cannot handle ragged tensors
            #paths_denom_log_weight_sum = tf.math.reduce_logsumexp(
            #    paths_denom_log_weights,
            #    axis=-1)

            # Log Transition weights for each "step" in path
            paths_nom_log_weight = tf.gather_nd(transition_weights, 
                                            tf.expand_dims(paths_nom, -1))
            
        else:
            if self._tw_absolute:
                transition_weights = tf.math.abs(transition_weights)

            paths_denom_log_weight_sum = tf.math.log(tf.reduce_sum(tf.gather_nd(transition_weights, tf.expand_dims(paths_denom, -1)), axis=-1))

            # Log Transition weights for each "step" in path
            paths_nom_log_weight = tf.math.log(tf.gather_nd(transition_weights, 
                                            tf.expand_dims(paths_nom, -1)))
            
        ### Paths logits to path variant probabilities
        # Paths logit
        path_logit = tf.reduce_sum(paths_nom_log_weight, axis=-1) - tf.reduce_sum(paths_denom_log_weight_sum, axis=-1)

        # Paths probabilities
        paths_probabilities = tf.math.exp(path_logit)

        # Variant probabilities (sum over associated paths)
        variant_probabilities = tf.reduce_sum(
            tf.gather_nd(paths_probabilities, tf.expand_dims(variant_2_paths, -1)), 
            axis=-1)

        #print(variant_probabilities)
        return variant_probabilities


    def _get_tw_initializer(self):
        if self._use_log_domain_weights:
            # Log domain values -> Real values approx (0.05, 2.7)
            w_transitions_init = tf.random_uniform_initializer(minval=-3, maxval=1)
        else:
            w_transitions_init = tf.random_uniform_initializer(minval=0.25, maxval=1.75)
        return w_transitions_init


    def _adapt_hot_start_weights(self, w_transitions_init: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
        w_transitions_init = super()._adapt_hot_start_weights(w_transitions_init)
        if self._use_log_domain_weights:
            return np.log(w_transitions_init)
        else:
            return w_transitions_init

    ################################################################################
    # Properties
    ################################################################################
    @property
    def transition_weights(self):
        if self._use_log_domain_weights:
            return np.exp(self.w_transitions.numpy())
        else:
            if self._tw_absolute:
                return np.abs(self.w_transitions.numpy())
            else:
                return self.w_transitions.numpy()
