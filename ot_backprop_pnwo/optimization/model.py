import numpy as np
import numpy.typing as npt
import tensorflow as tf
import logging
from enum import Enum, auto
from typing import Union

from ot_backprop_pnwo.optimization.model_layers import Path2VariantProbabilityLayer, Path2VariantProbabilityLayerBase, Path2VariantProbabilityLayerLogDomain

class ResidualHandling(str, Enum):
    ADD_RESIDUAL_ELEMENT = 'ADD_RESIDUAL_ELEMENT'
    NORMALIZE = 'NORMALIZE'


class Path2VariantLayerTypes(str, Enum):
    BASE_ABS = 'BASE_ABS'
    BASE_REG_LOG_BARRIER = 'BASE_REG_LOG_BARRIER'
    BASE_CLIP = 'BASE_CLIP'
    EXP_LOG_ABS = 'EXP_LOG_ABS'
    EXP_LOG_CLIP = 'EXP_LOG_CLIP'
    LOG_DOMAIN = 'LOG_DOMAIN'


class Path2VariantProbabilityModel(tf.keras.Model):

    def __init__(self, path_variant_layer:Path2VariantProbabilityLayer, anticipate_path_prob_residual: bool, residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT):
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._path_variant_layer = path_variant_layer
        self._anticipate_path_prob_residual = anticipate_path_prob_residual
        self._residual_handling=residual_handling


    def call(self, inputs):
        prob_variants = self._path_variant_layer(inputs)

        probabilities_out = prob_variants
        # Add artificial model path for residual probability, if there might be a residual AND we do not normalize
        # This assumes that the cost matrix that is used later also has an additional row!
        if (self._residual_handling == ResidualHandling.ADD_RESIDUAL_ELEMENT) and self._anticipate_path_prob_residual:
            self._logger.debug("Adding residual mock path to model")
            residual_probability = 1 - tf.reduce_sum(probabilities_out, axis=-1, keepdims=True)
            probabilities_out = tf.concat([probabilities_out, residual_probability], 0) 

        # !!!!!!!!!! IMPORTANT !!!!!!!!!!
        # The following operations can hide misconfigurations. 
        # For example, if no residual paths are added though needed, this will not raise any error anymore.
        # We assume that the probabilities returned would be properly normalized given that
        # we would compute with inifine precision.
        # Since this is not the case:
        # - Ensure positive probabilities
        # - Re-normalize due to lost precision
        # Ensure all probabilities positive
        probabilities_out = tf.clip_by_value(probabilities_out, 0, 10000)
        # Casting causes problems during the backward pass
        #probabilities_out = tf.cast(probabilities_out, tf.float64)
        probabilities_out = tf.divide(probabilities_out, tf.reduce_sum(probabilities_out, axis=-1, keepdims=True))
        return probabilities_out

    @property
    def adds_extra_losses(self):
        return self._path_variant_layer.adds_extra_losses

    @property
    def transition_weights(self):
        return self._path_variant_layer.transition_weights


class Path2VariantModelFactory:

    @staticmethod
    def init_configuration_with_default(w_transitions_init:  Union[npt.NDArray[np.float_], None]=None, 
                                nbr_transitions: Union[int, None]=None, 
                                anticipate_path_prob_residual: bool=True,
                                residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT,
                                layer_type: Path2VariantLayerTypes=Path2VariantLayerTypes.BASE_ABS
                            ) -> Path2VariantProbabilityModel:
        """ Factory methods that instantiates a paths to variants' probability model based on a confiuration name using default parameters if that configuration uses parameters

        Args:
            w_transitions_init (npt.NDArray[np.float_]|None): Initialization weights for transitions. Either it is specified or the number of transitions must be provided. Default to None.
            nbr_transitions (int|None): Number of transitions. Either it is specified or hot start initialization must be provided. Default to None.
            anticipate_path_prob_residual (bool): Anticipate that there might be residual probability on the side of the path sample. EMD assumes equal weight for both sides; thus, the residual with probability 1 - (sum of probability of paths) needs to be handled. Defaults to True.
            residual_handling (ResidualHandling): In case there is residual probability mass for the paths, this defines how it is handled. Defaults to adding a residual path.
            layer_type (Path2VariantLayerTypes): Which implementation of the [(weights + path sample) -> variant probability] to use (e.g., absolute weights, log-domain, clipping, ...)
        """
        if layer_type is Path2VariantLayerTypes.BASE_ABS:
            return Path2VariantModelFactory.init_base_model_abs(w_transitions_init, nbr_transitions, anticipate_path_prob_residual, residual_handling=residual_handling)
        elif layer_type is Path2VariantLayerTypes.BASE_REG_LOG_BARRIER:
            return Path2VariantModelFactory.init_base_model_barrier_reg(w_transitions_init, nbr_transitions, anticipate_path_prob_residual, residual_handling=residual_handling)
        elif layer_type is Path2VariantLayerTypes.BASE_CLIP:
            return Path2VariantModelFactory.init_base_model_clip(w_transitions_init, nbr_transitions, anticipate_path_prob_residual, residual_handling=residual_handling)
        elif layer_type is Path2VariantLayerTypes.EXP_LOG_ABS:
            return Path2VariantModelFactory.init_log_model_abs(w_transitions_init, nbr_transitions, anticipate_path_prob_residual, residual_handling=residual_handling)
        elif layer_type is Path2VariantLayerTypes.EXP_LOG_CLIP:
            return Path2VariantModelFactory.init_log_model_clip(w_transitions_init, nbr_transitions, anticipate_path_prob_residual, residual_handling=residual_handling)
        elif layer_type is Path2VariantLayerTypes.LOG_DOMAIN:
            return Path2VariantModelFactory.init_log_domain_model(w_transitions_init, nbr_transitions, anticipate_path_prob_residual, residual_handling=residual_handling)
        else:
            raise Exception("Unkown path-to-variant layer type")

    @staticmethod
    def init_base_model_abs(w_transitions_init:  Union[npt.NDArray[np.float_], None]=None, 
                                nbr_transitions: Union[int, None]=None, 
                                anticipate_path_prob_residual: bool=True,
                                residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT
                            ) -> Path2VariantProbabilityModel:
        """ Factory methods that instantiates a paths to variants' probability model that uses ABSOLUTE transition weights to ensure their non-negativity

        Args:
            w_transitions_init (npt.NDArray[np.float_]|None): Initialization weights for transitions. Either it is specified or the number of transitions must be provided. Default to None.
            nbr_transitions (int|None): Number of transitions. Either it is specified or hot start initialization must be provided. Default to None.
            anticipate_path_prob_residual (bool): Anticipate that there might be residual probability on the side of the path sample. EMD assumes equal weight for both sides; thus, the residual with probability 1 - (sum of probability of paths) needs to be handled. Defaults to True.
            residual_handling (ResidualHandling): In case there is residual probability mass for the paths, this defines how it is handled. Defaults to adding a residual path.
        """
        path_variant_layer = Path2VariantProbabilityLayerBase(w_transitions_init, nbr_transitions, tw_absolute=True)
        path_variant_model = Path2VariantModelFactory._compose_model(path_variant_layer, anticipate_path_prob_residual=anticipate_path_prob_residual, residual_handling=residual_handling)
        return path_variant_model
    
    @staticmethod
    def init_base_model_barrier_reg(w_transitions_init:  Union[npt.NDArray[np.float_], None]=None,
                                        nbr_transitions: Union[int, None]=None, 
                                        anticipate_path_prob_residual: bool=True,
                                        residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT,
                                        tw_log_barrier_mu: Union[float, None]=0.0007,
                                        tw_regularize_f: Union[float, None]=0.001
                                    ) -> Path2VariantProbabilityModel:
        """ Factory methods that instantiates a paths to variants' probability model that adds log barrier loss terms and weight regularization to ensure non-negative (and bounded) transition weights.

        Args:
            w_transitions_init (npt.NDArray[np.float_]|None): Initialization weights for transitions. Either it is specified or the number of transitions must be provided. Default to None.
            nbr_transitions (int|None): Number of transitions. Either it is specified or hot start initialization must be provided. Default to None.
            anticipate_path_prob_residual (bool): Anticipate that there might be residual probability on the side of the path sample. EMD assumes equal weight for both sides; thus, the residual with probability 1 - (sum of probability of paths) needs to be handled. Defaults to True.
            residual_handling (ResidualHandling): In case there is residual probability mass for the paths, this defines how it is handled. Defaults to adding a residual path.
            t_weights_barrier_mu (float|None): Weight of the logarithmic barrier term loss. If none, no barrier loss will be added. Defaults to 0.0007.
            t_weights_reg_f (float|None): Weight of the transition weight regularization loss. If none, no weight regularization loss will be added. Defaults to 0.001.
        """
        path_variant_layer = Path2VariantProbabilityLayerBase(
            w_transitions_init=w_transitions_init,
            nbr_transitions=nbr_transitions,
            tw_absolute=False,
            tw_log_barrier_mu=tw_log_barrier_mu,
            tw_regularize_f=tw_regularize_f, 
            tw_clip_min=None, tw_clip_max=None)
        path_variant_model = Path2VariantModelFactory._compose_model(path_variant_layer, anticipate_path_prob_residual=anticipate_path_prob_residual, residual_handling=residual_handling)
        return path_variant_model

    @staticmethod
    def init_base_model_clip(w_transitions_init:  Union[npt.NDArray[np.float_], None]=None,
                                    nbr_transitions: Union[int, None]=None, 
                                    anticipate_path_prob_residual: bool=True,
                                    residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT,
                                    tw_clip_min=0.00001, tw_clip_max=50
                            ) -> Path2VariantProbabilityModel:
        """ Factory methods that instantiates a paths to variants' probability model that clips transition weights to ensure their non-negativity.

        Args:
            w_transitions_init (npt.NDArray[np.float_]|None): Initialization weights for transitions. Either it is specified or the number of transitions must be provided. Default to None.
            nbr_transitions (int|None): Number of transitions. Either it is specified or hot start initialization must be provided. Default to None.
            anticipate_path_prob_residual (bool): Anticipate that there might be residual probability on the side of the path sample. EMD assumes equal weight for both sides; thus, the residual with probability 1 - (sum of probability of paths) needs to be handled. Defaults to True.
            residual_handling (ResidualHandling): In case there is residual probability mass for the paths, this defines how it is handled. Defaults to adding a residual path.
            t_weights_clip_min (float): Minimum value to which transitions weights are clipped. If none, no clipping is applied. Defaults to 0.00001. 
            t_weights_clip_max (float): Maximum value to which transitions weights are clipped. If none, no clipping is applied. Defaults to 50.
        """

        path_variant_layer = Path2VariantProbabilityLayerBase(
            w_transitions_init=w_transitions_init,
            nbr_transitions=nbr_transitions,
            tw_absolute=False,
            tw_log_barrier_mu=None,
            tw_regularize_f=None, 
            tw_clip_min=tw_clip_min, tw_clip_max=tw_clip_max)
        path_variant_model = Path2VariantModelFactory._compose_model(path_variant_layer, anticipate_path_prob_residual=anticipate_path_prob_residual, residual_handling=residual_handling)
        return path_variant_model

    @staticmethod
    def init_log_model_abs(w_transitions_init:  Union[npt.NDArray[np.float_], None]=None, 
                                nbr_transitions: Union[int, None]=None, 
                                anticipate_path_prob_residual: bool=True,
                                residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT
                            ) -> Path2VariantProbabilityModel:
        """ Factory methods that instantiates a paths to variants' probability model that uses ABSOLUTE transition weights to ensure their non-negativity

        The model employs a exp(log(...)) trick to the paths' probability computation to make it more robust and faster (avoid product rule).

        Args:
            w_transitions_init (npt.NDArray[np.float_]|None): Initialization weights for transitions. Either it is specified or the number of transitions must be provided. Default to None.
            nbr_transitions (int|None): Number of transitions. Either it is specified or hot start initialization must be provided. Default to None.
            anticipate_path_prob_residual (bool): Anticipate that there might be residual probability on the side of the path sample. EMD assumes equal weight for both sides; thus, the residual with probability 1 - (sum of probability of paths) needs to be handled. Defaults to True.
            residual_handling (ResidualHandling): In case there is residual probability mass for the paths, this defines how it is handled. Defaults to adding a residual path.
        """

        path_variant_layer = Path2VariantProbabilityLayerLogDomain(w_transitions_init, nbr_transitions, tw_absolute=True)
        path_variant_model = Path2VariantModelFactory._compose_model(path_variant_layer, anticipate_path_prob_residual=anticipate_path_prob_residual, residual_handling=residual_handling)
        return path_variant_model

    @staticmethod
    def init_log_model_clip(w_transitions_init:  Union[npt.NDArray[np.float_], None]=None,
                                nbr_transitions: Union[int, None]=None, 
                                anticipate_path_prob_residual: bool=True,
                                residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT,
                                tw_clip_min=0.00001, tw_clip_max=50
                            ) -> Path2VariantProbabilityModel:
        """ Factory methods that instantiates a paths to variants' probability model that clips transition weights to ensure their non-negativity.

        The model employs a exp(log(...)) trick to the paths' probability computation to make it more robust and faster (avoid product rule).

        Args:
            w_transitions_init (npt.NDArray[np.float_]|None): Initialization weights for transitions. Either it is specified or the number of transitions must be provided. Default to None.
            nbr_transitions (int|None): Number of transitions. Either it is specified or hot start initialization must be provided. Default to None.
            anticipate_path_prob_residual (bool): Anticipate that there might be residual probability on the side of the path sample. EMD assumes equal weight for both sides; thus, the residual with probability 1 - (sum of probability of paths) needs to be handled. Defaults to True.
            residual_handling (ResidualHandling): In case there is residual probability mass for the paths, this defines how it is handled. Defaults to adding a residual path.
            t_weights_clip_min (float): Minimum value to which transitions weights are clipped. If none, no clipping is applied. Defaults to 0.00001. 
            t_weights_clip_max (float): Maximum value to which transitions weights are clipped. If none, no clipping is applied. Defaults to 50.
        """
        path_variant_layer = Path2VariantProbabilityLayerLogDomain(
            w_transitions_init=w_transitions_init,
            nbr_transitions=nbr_transitions,
            log_domain=False,
            tw_absolute=False,
            tw_clip_min=tw_clip_min, tw_clip_max=tw_clip_max)
        path_variant_model = Path2VariantModelFactory._compose_model(path_variant_layer, anticipate_path_prob_residual=anticipate_path_prob_residual, residual_handling=residual_handling)
        return path_variant_model

    @staticmethod
    def init_log_domain_model(w_transitions_init:  Union[npt.NDArray[np.float_], None]=None,
                                nbr_transitions: Union[int, None]=None, 
                                anticipate_path_prob_residual: bool=True,
                                residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT,
                            ) -> Path2VariantProbabilityModel:
        """ Factory methods that instantiates a paths to variants' probability model that uses the logits of the transition weights during the complete computation.

        Computations involving transition weights are performed on their logits.
        This natively impedes negative transition weights.
        Querying the weights returns the exponentiated current transition weights. 

        Args:
            w_transitions_init (npt.NDArray[np.float_]|None): Initialization weights for transitions. Either it is specified or the number of transitions must be provided. Default to None.
            nbr_transitions (int|None): Number of transitions. Either it is specified or hot start initialization must be provided. Default to None.
            anticipate_path_prob_residual (bool): Anticipate that there might be residual probability on the side of the path sample. EMD assumes equal weight for both sides; thus, the residual with probability 1 - (sum of probability of paths) needs to be handled. Defaults to True.
            residual_handling (ResidualHandling): In case there is residual probability mass for the paths, this defines how it is handled. Defaults to adding a residual path.
        """
        path_variant_layer = Path2VariantProbabilityLayerLogDomain(
            w_transitions_init=w_transitions_init,
            nbr_transitions=nbr_transitions,
            log_domain=True,
            tw_absolute=False,
            tw_clip_min=None, tw_clip_max=None)
        path_variant_model = Path2VariantModelFactory._compose_model(path_variant_layer, anticipate_path_prob_residual=anticipate_path_prob_residual, residual_handling=residual_handling)
        return path_variant_model

    @staticmethod
    def _compose_model(path_variant_layer: Path2VariantProbabilityLayer, anticipate_path_prob_residual=True, residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT) -> Path2VariantProbabilityModel:
        """ Compose the model given the path-to-variant-probability layer

        Args:
            path_variant_layer (Path2VariantProbability): Instance of a path-to-variant-probability layer
            anticipate_path_prob_residual (bool): Anticipate that there might be residual probability on the side of the path sample. EMD assumes equal weight for both sides; thus, the residual with probability 1 - (sum of probability of paths) needs to be handled. Defaults to True.
            residual_handling (ResidualHandling): In case there is residual probability mass for the paths, this defines how it is handled. Defaults to adding a residual path.
        """
        path_variant_model = Path2VariantProbabilityModel(path_variant_layer, anticipate_path_prob_residual=anticipate_path_prob_residual, residual_handling=residual_handling)
        path_variant_model = Path2VariantModelFactory._post_process_model(path_variant_model)
        return path_variant_model

    @staticmethod
    def _post_process_model(path_variant_model: Path2VariantProbabilityModel):
        """ Post-process the model

        """
        # Extra losses caused problems when using graph mode.
        # Otherwise, enforce graph mode.
        # !!! If I'm not mistaking, this not automatically the cases when using a custom training loop !!!
        if not path_variant_model.adds_extra_losses:
            path_variant_model.call = tf.function(path_variant_model.call, reduce_retracing=True)
            #path_variant_model.call = tf.function(path_variant_model.call)
        return path_variant_model
