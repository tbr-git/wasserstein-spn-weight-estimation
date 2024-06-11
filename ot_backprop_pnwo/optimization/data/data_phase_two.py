import logging
import tensorflow as tf
import numpy as np
import abc 
from dataclasses import dataclass
from typing import Union

from ot_backprop_pnwo.optimization.data.data_creation import create_dataset_flex_paths, create_dataset_flex_paths_variants, create_dataset_flex_variants, spn_path_language_to_tensors
from ot_backprop_pnwo.optimization.data.data_phase_one import DataPhaseOne
from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType
from ot_backprop_pnwo.spn.spn_path_sampling import SPN_TRANSITION_PROBABILITY
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang

"""
Data used in each iteration (gradient descent step) of the method.

Depending on whether the spn was fully unfolded or the entire log is considered,
certain inputs are fixed (i.e., always the same) (even if inputs change, they might be tf.constants!).

"""
@dataclass
class DataPhaseTwoIteration:
    # Stochastic path representation using tensorflow datatypes
    tf_variant_2_paths: tf.RaggedTensor
    tf_paths_nom: tf.RaggedTensor
    tf_paths_denom: tf.RaggedTensor

    # Cost matrix
    cost_matrix: tf.Tensor
    # Variant likelihoods
    variant_lh: tf.Tensor

"""
    Data for the second phase of "Almost End-2-end SPN Weight Estimation" using OT.
    
    Base container.
"""
class DataPhaseTwo(metaclass=abc.ABCMeta):
    def __init__(self, spn_container: SPNWrapper, act_id_conn: ActivityIDConnector, 
                 is_model_input_fixed: bool, is_log_side_fixed: bool): 
        self._spn_container = spn_container
        self._act_id_conn = act_id_conn
        self._is_model_input_fixed = is_model_input_fixed
        self._is_log_side_fixed =  is_log_side_fixed

    @property
    def spn_container(self):
        return self._spn_container

    @property
    def act_id_conn(self):
        return self._act_id_conn

    @property
    def is_model_input_fixed(self) -> bool:
        return self._is_model_input_fixed

    @property
    def is_log_side_fixed(self) -> bool:
        return self._is_log_side_fixed

    @abc.abstractmethod
    def get_next_iteration_data(self) -> DataPhaseTwoIteration:
        pass


"""
    Data for the second phase of "Almost End-2-end SPN Weight Estimation" using OT.
    
    Container for
    - Flexible paths (due to large number of paths in SPN)
    - Flexible variants (due to large number of variant in log)
"""
class DataPhaseTwoFlexPFlexL(DataPhaseTwo):

    def __init__(self, spn_container: SPNWrapper, act_id_conn: ActivityIDConnector, 
            tf_data_flex_path_flex_log: tf.data.Dataset):
        super().__init__(spn_container, act_id_conn, False, False)
        self._tf_data_flex_path_flex_log = tf_data_flex_path_flex_log

        # Repeat the dataset
        self._tf_data_flex_path_flex_log = self._tf_data_flex_path_flex_log.repeat()
        # TF Iterator
        self._iterator_tf_data = iter(self._tf_data_flex_path_flex_log)

    def get_next_iteration_data(self) -> DataPhaseTwoIteration:
        (tf_variant_2_paths, tf_paths_nom, tf_paths_denom, tf_b, tf_C) = self._iterator_tf_data.get_next()
        return DataPhaseTwoIteration(tf_variant_2_paths, tf_paths_nom, tf_paths_denom, tf_C, tf_b)


class DataPhaseTwoFlexPFixL(DataPhaseTwo):
    
    def __init__(self, spn_container: SPNWrapper, act_id_conn: ActivityIDConnector, 
            tf_data_flex_path: tf.data.Dataset, variant_lh: np.ndarray[np.float32]):
        super().__init__(spn_container, act_id_conn, False, True)
        self._tf_data_flex_path = tf_data_flex_path
        self._variant_lh = tf.constant(variant_lh)

        # Repeat the dataset
        self._tf_data_flex_path = self._tf_data_flex_path.repeat()
        # TF Iterator
        self._iterator_tf_data = iter(self._tf_data_flex_path)

    def get_next_iteration_data(self) -> DataPhaseTwoIteration:
        (tf_variant_2_paths, tf_paths_nom, tf_paths_denom, tf_C) = self._iterator_tf_data.get_next()
        return DataPhaseTwoIteration(tf_variant_2_paths, tf_paths_nom, tf_paths_denom, tf_C, self._variant_lh)


class DataPhaseTwoFixPFlexL(DataPhaseTwo):
    
    def __init__(self, spn_container: SPNWrapper, act_id_conn: ActivityIDConnector, 
            tf_variant_2_paths: tf.RaggedTensor, tf_paths_nom: tf.RaggedTensor, tf_paths_denom: tf.RaggedTensor,
            tf_data_flex_variants: tf.data.Dataset):
        super().__init__(spn_container, act_id_conn, False, True)
        # Log side
        self._tf_data_flex_variants = tf_data_flex_variants
        # SPN side
        self._tf_variant_2_paths = tf_variant_2_paths
        self._tf_paths_nom = tf_paths_nom
        self._tf_paths_denom = tf_paths_denom

        # Repeat the dataset
        self._tf_data_flex_variants = self._tf_data_flex_variants.repeat()
        # TF Iterator
        self._iterator_tf_data = iter(self._tf_data_flex_variants)

    def get_next_iteration_data(self) -> DataPhaseTwoIteration:
        (tf_variant_llh, tf_C) = self._iterator_tf_data.get_next()
        return DataPhaseTwoIteration(self._tf_variant_2_paths, self._tf_paths_nom, self._tf_paths_denom, tf_C, tf_variant_llh)


class DataPhaseTwoFactory:

    def create_data_phase_two(spn_container: SPNWrapper, act_id_conn: ActivityIDConnector, 
                              stoch_lang_log: StochasticLang, 
                              data_phase_one: DataPhaseOne,
                              nbr_paths: int, nbr_variants: int, nbr_samples: int,
                              emsc_loss_type: EMSCLossType,
                              ):
        logger = logging.getLogger(DataPhaseTwoFactory.__class__.__name__)

        # Distinguish flexible path and flexible log input
        # Distinction required to created dedicated training regimes
        if data_phase_one.is_spn_unfolded and data_phase_one.is_log_complete:
            logger.info("Phase two not required. Will not build data.")
            return None
        elif data_phase_one.is_spn_unfolded and not data_phase_one.is_log_complete:
            logger.info("Creating data phase 2: SPN completely unfolder | Multiple log samples")
            return DataPhaseTwoFactory._create_data_fix_flex(
                data_phase_one, stoch_lang_log, nbr_variants, nbr_samples, emsc_loss_type)
        elif not data_phase_one.is_spn_unfolded and data_phase_one.is_log_complete:
            logger.info("Creating data phase 2: Multiple path samples | Full log")
            return DataPhaseTwoFactory._create_data_flex_fix(spn_container, act_id_conn, stoch_lang_log, 
                                                              nbr_paths, nbr_variants, nbr_samples, emsc_loss_type)
        elif not data_phase_one.is_spn_unfolded and not data_phase_one.is_log_complete:
            logger.info("Creating data phase 2: Multiple path samples | Multiple log samples")
            return DataPhaseTwoFactory._create_data_flex_flex(spn_container, act_id_conn, stoch_lang_log, 
                                                              nbr_paths, nbr_variants, nbr_samples, emsc_loss_type)
        else:
            raise Exception("Ups! Unforseen and not implemented scenario for phase two input")
        
    def _create_data_flex_flex(spn_container: SPNWrapper, act_id_conn: ActivityIDConnector, 
                               stoch_lang_log: StochasticLang, 
                               nbr_paths: int, nbr_variants: int, nbr_samples: int,
                               emsc_loss_type: EMSCLossType,
                               ) -> DataPhaseTwoFlexPFlexL:
        tf_data = create_dataset_flex_paths_variants(spn_container, stoch_lang_log, 
                                                act_id_conn, nbr_samples, 
                                                nbr_paths, nbr_variants, emsc_loss_type, trans_lh=SPN_TRANSITION_PROBABILITY.TRANSITION_WEIGHT)
        return DataPhaseTwoFlexPFlexL(spn_container, act_id_conn=act_id_conn, tf_data_flex_path_flex_log=tf_data)

    def _create_data_flex_fix(spn_container: SPNWrapper, act_id_conn: ActivityIDConnector, 
                               stoch_lang_log: StochasticLang, 
                               nbr_paths: int, nbr_variants: int, nbr_samples: int,
                               emsc_loss_type: EMSCLossType,
                               ) -> DataPhaseTwoFlexPFixL:
        logger = logging.getLogger(DataPhaseTwoFactory.__class__.__name__)
        log_lang_tmp = stoch_lang_log
        is_log_complete = True
        if len(log_lang_tmp) > nbr_variants:
            is_log_complete = False
            logger.info("Training with fixed log language but support of log lanugage is too large -> Sampling")
            (sl_sample_variants, sl_sample_prob) = log_lang_tmp.most_likely_variants(nbr_variants)
            # Pseudo language (might not be normalized)
            log_lang_tmp = StochasticLang(act_id_conn, sl_sample_variants, sl_sample_prob)

        (tf_data, variant_lh) = create_dataset_flex_paths(spn_container, log_lang_tmp,
                                                    act_id_conn, nbr_samples, nbr_paths, 
                                                    emsc_loss_type,
                                                    add_log_residual=not is_log_complete,
                                                    trans_lh=SPN_TRANSITION_PROBABILITY.TRANSITION_WEIGHT) 

        return DataPhaseTwoFlexPFixL(spn_container, act_id_conn, tf_data, variant_lh)

    def _create_data_fix_flex(
            data_phase_one: DataPhaseOne,
            stoch_lang_log: StochasticLang, 
            nbr_variants: int, 
            nbr_samples: int,
            emsc_loss_type: EMSCLossType,
        ) -> DataPhaseTwoFlexPFixL:
        tf_data_flex_variants = create_dataset_flex_variants(data_phase_one.pseudo_spn_lang, stoch_lang_log,
                                                data_phase_one.act_id_conn, nbr_samples, 
                                                nbr_variants, not data_phase_one.is_spn_unfolded, 
                                                emsc_loss_type)
                                            
        # Could share it with first phase, but just create some fresh nodes (for safety)
        # Will avoid entangled computational graphs, yet might also be unnecessary, I don't know
        ((tf_variant_2_paths, tf_paths_nom, tf_paths_denom), stoch_lang_spn) = spn_path_language_to_tensors(
            data_phase_one.spn_paths, data_phase_one.act_id_conn
        )
        return DataPhaseTwoFixPFlexL(
            spn_container=data_phase_one.spn_container,
            act_id_conn=data_phase_one.act_id_conn,
            tf_data_flex_variants= tf_data_flex_variants,
            tf_paths_nom=tf_paths_nom,
            tf_paths_denom=tf_paths_denom,
            tf_variant_2_paths=tf_variant_2_paths)

        
