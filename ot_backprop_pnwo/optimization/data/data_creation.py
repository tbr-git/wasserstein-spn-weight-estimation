from collections import namedtuple
from functools import partial
import logging
import tensorflow as tf
import numpy as np

from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType
from ot_backprop_pnwo.spn.spn_path_sampling import SPN_TRANSITION_PROBABILITY, get_robust_path_sample, log_sample_statistics
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.spn.stochastic_path import StochasticPath
from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang, lvs_cost_matrix

logger = logging.getLogger(__name__)


def prepare_cost_matrix_log_probabilities(stoch_lang_spn: StochasticLang, pseudo_stoch_lang_log: StochasticLang, 
                                          add_path_residual_term=True, add_log_residual_term=True,
                                          emsc_loss_type=EMSCLossType.PEMSC
                                          ):
    logger.debug("Computing cost matrix.")

    C = lvs_cost_matrix(stoch_lang_spn, pseudo_stoch_lang_log)

    # Add row for "residual" path
    if add_path_residual_term:
        logger.debug("Adding residual model trace to cost matrix.")
        if emsc_loss_type == EMSCLossType.PEMSC:
            C = np.concatenate((C, np.ones((1, len(pseudo_stoch_lang_log)))), axis=0)
            logger.debug("Adding residual model trace for PEMSC.")
        elif emsc_loss_type == EMSCLossType.EMSC:
            C = np.concatenate((C, np.min(C, axis=0)[np.newaxis, :]), axis=0)
            logger.debug("Adding residual model trace for EMSC.")
        else:
            raise Exception("Unkown EMSC Loss type, not clear how to add the residual path")

    log_probabilities = pseudo_stoch_lang_log.probabilities
    # Log language might not be normalized
    # Add virtual pseudo variant and adapt cost matrix
    if add_log_residual_term:
        res_prob_log = 1 - np.sum(pseudo_stoch_lang_log.probabilities)
        log_probabilities = pseudo_stoch_lang_log.probabilities
        logger.debug("Adding residual log trace to cost matrix.")
        log_probabilities = np.append(log_probabilities, res_prob_log)
        # extra column for the residual path row
        len_spn_lang = len(stoch_lang_spn) + (1 if add_path_residual_term else 0)

        if emsc_loss_type == EMSCLossType.PEMSC:
            # Distance 1 to every path (except for 0 for the residual path)
            C = np.concatenate((C, np.ones((len_spn_lang, 1))), axis=1)
            logger.debug("Adding residual trace for PEMSC.")
        elif emsc_loss_type == EMSCLossType.EMSC:
            C = np.concatenate((C, np.min(C, axis=1)[:, np.newaxis]), axis=1)
            logger.debug("Adding residual trace for EMSC.")
        else:
            raise Exception("Unkown EMSC Loss type, not clear how to add the residual path")
        if add_path_residual_term:
            C[-1, -1] = 0
    return (log_probabilities, C)


def create_dataset_flex_paths_variants(spn_container: SPNWrapper, stoch_lang_log: StochasticLang, 
                   act_id_conn: ActivityIDConnector, nbr_sampling_runs: int, 
                   sample_size_model: int, sample_size_log: int, 
                   emsc_loss_type: EMSCLossType,
                   trans_lh: SPN_TRANSITION_PROBABILITY=SPN_TRANSITION_PROBABILITY.UNIFORM, 
                   ):
    l_tf_variant_2_paths = []
    l_tf_paths_nom = []
    l_tf_paths_denom = []
    l_tf_variant_prob = []
    l_tf_C = []

    logger.debug("Creating dataset of path and variant samples")

    spn_paths_samples = get_robust_path_sample(spn_container=spn_container, 
                                               trans_lh=trans_lh, 
                                               sample_size=sample_size_model, 
                                               nbr_samples=nbr_sampling_runs)
    log_sample_statistics(spn_paths_samples)

    for s_run in range(nbr_sampling_runs):
        stoch_paths = spn_paths_samples[s_run]
        # Sample variants
        (sl_sample_variants, sl_sample_prob) = stoch_lang_log.random_subsample(sample_size_log)

        stoch_lang_spn, net_lang_2_path = StochasticLang.from_stochastic_paths(
            stoch_paths, act_id_conn)

        ####################
        # Cost Matrix
        ####################
        (b, C) = prepare_cost_matrix_log_probabilities(stoch_lang_spn, 
                StochasticLang(act_id_conn, sl_sample_variants, sl_sample_prob),
                add_path_residual_term=True, add_log_residual_term=True, emsc_loss_type=emsc_loss_type)

        # Create tensors
        tf_paths_nom = tf.ragged.constant(list([p.transition_ids for p in stoch_paths]), dtype=tf.int32)
        tf_paths_denom = tf.ragged.constant(list([p.curr_enabled_transitions for p in stoch_paths]), dtype=tf.int32)
        tf_variant_2_paths = tf.ragged.constant(
            tuple(net_lang_2_path[i] for i in range(len(net_lang_2_path))),
            dtype=tf.int32
        )
        tf_b = tf.constant(b, dtype=tf.float32)
        tf_C = tf.ragged.constant(C, dtype=tf.float32)

        l_tf_variant_2_paths.append(tf_variant_2_paths)
        l_tf_paths_nom.append(tf_paths_nom)
        l_tf_paths_denom.append(tf_paths_denom)
        l_tf_variant_prob.append(tf_b)
        l_tf_C.append(tf_C)
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.stack(l_tf_variant_2_paths), 
            tf.stack(l_tf_paths_nom), 
            tf.stack(l_tf_paths_denom), 
            tf.stack(l_tf_variant_prob),
            tf.stack(l_tf_C)
        ))
    return dataset


def create_dataset_flex_paths(spn_container: SPNWrapper, pseudo_stoch_lang_log: StochasticLang,
                   act_id_conn: ActivityIDConnector, nbr_sampling_runs: int, 
                   sampling_run_size: int, 
                   emsc_loss_type: EMSCLossType,
                   add_log_residual: bool=False, 
                   trans_lh: SPN_TRANSITION_PROBABILITY=SPN_TRANSITION_PROBABILITY.UNIFORM):

    l_tf_variant_2_paths = []
    l_tf_paths_nom = []
    l_tf_paths_denom = []
    l_tf_C = []
    logger.debug("Creating dataset of path samples")
    spn_paths_samples = get_robust_path_sample(spn_container=spn_container, 
                                               trans_lh=trans_lh, 
                                               sample_size=sampling_run_size, 
                                               nbr_samples=nbr_sampling_runs)
    log_sample_statistics(spn_paths_samples)

    for s_run in range(nbr_sampling_runs):
        stoch_paths = spn_paths_samples[s_run]
        
        stoch_lang_spn, net_lang_2_path = StochasticLang.from_stochastic_paths(stoch_paths, act_id_conn)
        (b, C) = prepare_cost_matrix_log_probabilities(stoch_lang_spn, 
                pseudo_stoch_lang_log,
                add_path_residual_term=True, add_log_residual_term=add_log_residual, emsc_loss_type=emsc_loss_type)
        
        # Create tensors
        tf_paths_nom = tf.ragged.constant(list([p.transition_ids for p in stoch_paths]), dtype=tf.int32)
        tf_paths_denom = tf.ragged.constant(list([p.curr_enabled_transitions for p in stoch_paths]), dtype=tf.int32)
        tf_variant_2_paths = tf.ragged.constant(
            tuple(net_lang_2_path[i] for i in range(len(net_lang_2_path))),
            dtype=tf.int32
        )
        tf_C = tf.ragged.constant(C, dtype=tf.float32)

        l_tf_variant_2_paths.append(tf_variant_2_paths)
        l_tf_paths_nom.append(tf_paths_nom)
        l_tf_paths_denom.append(tf_paths_denom)
        l_tf_C.append(tf_C)
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.stack(l_tf_variant_2_paths), 
            tf.stack(l_tf_paths_nom), 
            tf.stack(l_tf_paths_denom), 
            tf.stack(l_tf_C)
        ))
    return (dataset, b)


def create_dataset_flex_variants(stoch_lang_spn: StochasticLang,
                                 stoch_lang_log: StochasticLang,
                                 act_id_conn: ActivityIDConnector, nbr_sampling_runs: int, 
                                 sampling_run_size: int,
                                 extend_for_residual_path: bool,
                                 emsc_loss_type: EMSCLossType
                                 ):

    l_tf_variant_prob = []
    l_tf_C = []
    logger.debug("Creating dataset of log samples")
    for s_run in range(nbr_sampling_runs):
        # Sample variants
        (sl_sample_variants, sl_sample_prob) = stoch_lang_log.random_subsample(sampling_run_size)
        
        ####################
        # Cost Matrix
        ####################
        (b, C) = prepare_cost_matrix_log_probabilities(stoch_lang_spn, 
                StochasticLang(act_id_conn, sl_sample_variants, sl_sample_prob),
                add_path_residual_term=extend_for_residual_path,
                add_log_residual_term=True, emsc_loss_type=emsc_loss_type)
        # Create tensors
        tf_b = tf.constant(b, dtype=tf.float32)
        tf_C = tf.constant(C, dtype=tf.float32)

        l_tf_variant_prob.append(tf_b)
        l_tf_C.append(tf_C)
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.stack(l_tf_variant_prob), 
            tf.stack(l_tf_C)
        ))
    return dataset


def spn_path_language_to_tensors(spn_paths: tuple[StochasticPath], act_id_conn: ActivityIDConnector):
    # Stochastic Language
    stoch_lang_spn, net_lang_2_path = StochasticLang.from_stochastic_paths(
        spn_paths, 
        act_id_conn)
    ##############################
    # Ragged Input Tensors
    ##############################
    # Series Nominators 
    tf_paths_nom = tf.ragged.constant(list([p.transition_ids for p in spn_paths]), dtype=tf.int32)
    # Series Denomiators 
    tf_paths_denom = tf.ragged.constant(list([p.curr_enabled_transitions for p in spn_paths]), 
                                        dtype=tf.int32)

    # for variant v: [p1, ...] paths
    tf_variant_2_paths = tf.ragged.constant(
        tuple(net_lang_2_path[i] for i in range(len(net_lang_2_path)))
    )

    ### Safe constant inputs
    return ((tf_variant_2_paths, tf_paths_nom, tf_paths_denom), stoch_lang_spn)

