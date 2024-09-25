import logging
from dataclasses import dataclass
import tensorflow as tf
import numpy as np

from ot_backprop_pnwo.optimization.data.data_creation import prepare_cost_matrix_log_probabilities, spn_path_language_to_tensors
from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType
from ot_backprop_pnwo.optimization.model import ResidualHandling
from ot_backprop_pnwo.spn import spn_path_sampling
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.spn.stochastic_path import StochasticPath
from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang

@dataclass
class DataPhaseOne:
    # SPN
    spn_container: SPNWrapper
    # activity connection
    act_id_conn: ActivityIDConnector

    # Background data
    spn_paths: tuple[StochasticPath]
    # (Pseudo - If fraction is extracted, is not normalized)
    pseudo_spn_lang: StochasticLang
    pseudo_log_lang: StochasticLang

    # Optimization Model
    # Stochastic path representation using tensorflow datatypes
    tf_variant_2_paths: tf.RaggedTensor
    tf_paths_nom: tf.RaggedTensor
    tf_paths_denom: tf.RaggedTensor

    # Cost matrix
    cost_matrix: np.ndarray[np.float32]
    # Variant likelihoods
    variant_lh: np.ndarray[np.float32]

    # Some additional information
    # Do we capture all behavior of the SPN and log? 
    is_spn_unfolded: bool
    is_log_complete: bool
    # Sampling the most likely paths can fail.  Sort of BFS that fails for ridiculous branching
    is_path_most_likely: bool
    

class DataPhaseOneFactory:

    def create_p1_data(spn_container: SPNWrapper, stoch_lang_log: StochasticLang, 
            act_id_conn: ActivityIDConnector,
            max_nbr_paths: int=300, max_nbr_variants: int=300,
            emsc_loss_type=EMSCLossType.PEMSC,
            residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT
            ) -> DataPhaseOne:
        logger = logging.getLogger(DataPhaseOneFactory.__class__.__name__)
        logger.debug(f"Creating data for first phase aiming for OT Size ({max_nbr_paths}, {max_nbr_variants})")

        ### Stochastic Path Language
        # Try to sample nbr paths + 1
        # Sampling prefers "short" paths
        path_sample_data = spn_path_sampling.get_single_robust_path_sample(spn_container=spn_container, sample_size=max_nbr_paths + 1, 
                                                         trans_lh=spn_path_sampling.SPN_TRANSITION_PROBABILITY.UNIFORM, 
                                                         max_time_till_fallback_ms=30_000)
        paths = path_sample_data.paths_robustified

        if path_sample_data.nbr_paths_full - path_sample_data.nbr_paths_robustified > 0:
            logger.debug(f"Removed {path_sample_data.nbr_paths_full - path_sample_data.nbr_paths_robustified} while robustifying path samples.")

        # Careful: Consider paths before robustifying (since robustification usually should only have an effect in case of loops, this might however not cause problems) 
        is_spn_unfolded = (not path_sample_data.probed_sampling) and path_sample_data.nbr_paths_full <= max_nbr_paths
        # Take paths most likely using uniform weights
        # -> Prefers paths that are short and have low branching complexity
        if not is_spn_unfolded:
            paths = paths[:max_nbr_paths]

        ### Stochastic Log Language
        is_log_complete = len(stoch_lang_log) <= max_nbr_variants
        # Pseudo language: If truncated will not be normalized
        pseudo_stoch_lang_log = stoch_lang_log
        if not is_log_complete:
            (sl_sample_variants, sl_sample_prob) = stoch_lang_log.most_likely_variants(max_nbr_variants)
            # Sampling resolution strategy
            if residual_handling == ResidualHandling.NORMALIZE:
                sl_sample_prob /= np.sum(sl_sample_prob)
            pseudo_stoch_lang_log = StochasticLang(act_id_conn, sl_sample_variants, sl_sample_prob)



        ((tf_variant_2_paths, tf_paths_nom, tf_paths_denom), stoch_lang_spn) = spn_path_language_to_tensors(paths, act_id_conn)
        # Only add a residual path, if there is residual AND residual elements enabled
        add_path_residual_term = (not is_spn_unfolded) and residual_handling == ResidualHandling.ADD_RESIDUAL_ELEMENT
        add_log_residual_term = (not is_log_complete) and residual_handling == ResidualHandling.ADD_RESIDUAL_ELEMENT
        # Log-Side + Cost matrix
        # Using numpy because we use a python loss
        (b, C) = prepare_cost_matrix_log_probabilities(stoch_lang_spn, pseudo_stoch_lang_log, 
                                                        add_path_residual_term=add_path_residual_term, add_log_residual_term=add_log_residual_term, 
                                                        emsc_loss_type=emsc_loss_type)

        return DataPhaseOne(spn_container=spn_container, act_id_conn=act_id_conn, 
                            spn_paths=paths, pseudo_spn_lang=stoch_lang_spn, pseudo_log_lang=pseudo_stoch_lang_log,
                            tf_variant_2_paths=tf_variant_2_paths, tf_paths_nom=tf_paths_nom, tf_paths_denom=tf_paths_denom, 
                            cost_matrix=C, variant_lh=b, is_spn_unfolded=is_spn_unfolded, is_log_complete=is_log_complete,
                            is_path_most_likely=not path_sample_data.probed_sampling)
