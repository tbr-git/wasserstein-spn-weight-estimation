from collections import namedtuple
import numpy as np
import ot
from ot_backprop_pnwo.optimization.data.data_phase_one import DataPhaseOne, DataPhaseOneFactory
from ot_backprop_pnwo.optimization.data.data_phase_two import DataPhaseTwo, DataPhaseTwoFactory
from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType
from ot_backprop_pnwo.optimization.model import Path2VariantLayerTypes, Path2VariantModelFactory
import tensorflow as tf
import logging
import time
from collections.abc import Callable

from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang

module_logger = logging.getLogger(__name__)

OT_WO_Result = namedtuple('OT_WO_Result', ['spn_weights', 'error_series_phase_one', 'training_time_phase_one', 
                                           'error_series_phase_two', 'training_time_phase_two', 'full_time'])



class OT_WO_Two_Phase:
    """
    Implementation of a two-phase approach for optimizing weights in a stochastic Petri net (SPN).

    Starting from a set of paths sampled from the SPN, we directly optimize a (penalized) EMSC score.
    To this end, we exploit that the solution of the dual EMD problem can be used to create a subgradient for
    the input siganture. 

    Using subgradient descent, we backpropagate this subgradient to the paths of the SPN and, subsequently, to its weights.
    In case the language of the SPN is large (or even infinite) or the logs contains a large number of variants, we propose a two-phase approach.

    Phase 1:
        Run optimization for a fixed number of paths and variants. To this end, we consider the most likely paths in the SPN (asuming uniform weights) 
        as well as the most likely variants in the log. 
        This gives preference to paths that are short and have (low branching complexity). 
        Usually, these paths correspond to variants that are likely to occur in practice. 

        Due to the penalization, it is important that we focus on paths, which should have high likelihood in the optimal solution.
        The penalization encourages solutions that avoid remaining probability on the SPN side at all costs.  
        Even if a path has low probability in the final solution, the corresponding variant is likeliy to be allocatable to a variant in the log
        with costs less than 1, which is better than taking the penalty.

    Phase 2:
        Given a good initial solution, we create multiple batches of input path and variants. 
        Here, we incorporate some randomization to incorporate less likely paths to get some additional learning signal.
        However, we still try to incorporate the likely paths (under the current weight) and variants to avoid high costs due to the penalization of residual probability.
        In case we sample both SPN and log, we want the sampled probabilitiy mass on both sides to be similar to avoid strong signal due to probability flow to residual nodes.
    """

    # If both phases are executed, reduce maximal number of steps for phase 1
    PHASE_ONE_WARMUP_STEP_REDUCTION_FACTOR = 0.25
    # If both phases are executed, relax convergence criterion
    PHASE_ONE_WARMUP_CONVERGENCE_RELAX_FACTOR = 5

    def __init__(self, 
                 spn_container: SPNWrapper, 
                 stoch_lang_log: StochasticLang, 
                 act_id_conn: ActivityIDConnector, 
                 hot_start: bool=False,
                 max_nbr_paths: int=300, max_nbr_variants: int=300, 
                 run_phase_two=False, 
                 nbr_samples_phase_two = 10,
                 clip_gradients: bool=False,
                 layer_type: Path2VariantLayerTypes=Path2VariantLayerTypes.BASE_ABS,
                 emsc_loss_type: EMSCLossType=EMSCLossType.PEMSC
            ) -> None:
        """Constructor

        Args:
            spn_container (SPNWrapper): SPN container
            stoch_lang_log (StochasticLang): Stochastic language of the log
            act_id_conn (ActivityIDConnector): Connecting activities to category codes 
            hot_start (bool): Hot start even phase 1 with the current weights of the SPN. Defaults to False.
            max_nbr_pathes (int, optional): Maximum number of stochastic paths used 
                for each gradient decent step . Defaults to 300.
            max_nbr_variants (int, optional): Maximum number of variants considered on the log side
                for each gradient decent step . Defaults to 300.
            run_phase_two (bool, optional): Run phase two of the optimization. 
                If SPN and log are "small", it defaults to running phase 1 until convergence. Defaults to False.
            nbr_samples_phase_two: Number of (SPN paths, log variants) combination to use as input.
            clip_gradients (bool): Clip the gradient to [-1, 1] before applying them. Defaults to False.
            layer_type (Path2VariantLayerTypes): Layer that is used to transform paths to variant probabilities (e.g., using absolute transition weights, exp(log(product along path)))
            emsc_loss_type (EMSCLossType): Optimize EMSC or penalized EMSC, which penalizes residual flow with cost 1. Default to penalized EMSC.
        """
        # Logger
        self._logger = logging.getLogger(self.__class__.__name__)

        # Data
        self._spn_container = spn_container
        self._stoch_lang_log = stoch_lang_log
        self._act_id_conn = act_id_conn
        # OT Size
        self._max_nbr_paths = max_nbr_paths
        self._max_nbr_variants = max_nbr_variants
        # Phase 1
        self._hot_start = hot_start
        # Phase 2
        self._run_phase_two_init = run_phase_two # Safe for logging
        self._run_phase_two = run_phase_two
        self._nbr_samples_phase_two  = nbr_samples_phase_two
        # Gradient clipping
        self._clip_gradients = clip_gradients
        # Loss Type
        self._emsc_loss_type = emsc_loss_type
        # Layer Type
        self._layer_type = layer_type

        ########################################
        # Initialize Phase 1 
        ########################################
        time_init_start = time.time()
        self._data_phase_1 = DataPhaseOneFactory.create_p1_data(
            spn_container = self._spn_container,
            stoch_lang_log = self._stoch_lang_log,
            act_id_conn = self._act_id_conn,
            max_nbr_paths = self._max_nbr_paths,
            max_nbr_variants = self._max_nbr_variants,
            emsc_loss_type=self._emsc_loss_type
        )

        ### Update requirement of phase two
        if self._run_phase_two and self._data_phase_1.is_spn_unfolded and self._data_phase_1.is_log_complete:
            self._logger.debug("Given the size limits, we can fully unfold the SPN and consider the entire log, phase two will not be needed!")
            self._run_phase_two = False

        self._path_variant_model = None
        if hot_start:
            self._path_variant_model = Path2VariantModelFactory.init_configuration_with_default(
                w_transitions_init=self._spn_container.initial_weights, 
                add_res_prob_path=not self._data_phase_1.is_spn_unfolded,
                layer_type=self._layer_type)
        else:
            nbr_transitions = len(self._spn_container.net.transitions)
            self._path_variant_model = Path2VariantModelFactory.init_configuration_with_default(
                nbr_transitions=nbr_transitions, 
                add_res_prob_path=not self._data_phase_1.is_spn_unfolded,
                layer_type=self._layer_type)
        self._time_full = time.time() - time_init_start
        self._logger.info(f"Initialized " + self.configuration_description)

    def optimize_weights(self, optimizer, nbr_iterations_min=50, nbr_iterations_max=5000, eps_convergence=0.0025):
        self._logger.info("Running Optimization")
        time_opt_start = time.time()
        ### Phase 1
        start_time_p1 = time.time()
        train_step_phase_1 = self._create_training_regime_fixed_paths_fixed_variants(
            optimizer=optimizer,
            data_phase_one=self._data_phase_1)
        
        nbr_iterations_max_phase_1 = nbr_iterations_max
        eps_conv_phase_1 = eps_convergence
        nbr_iterations_min_phase_1 = nbr_iterations_min
        # If phase two is executed, relax convergence
        if self._run_phase_two:
            nbr_iterations_min_phase_1 = 50
            nbr_iterations_max_phase_1 = OT_WO_Two_Phase.PHASE_ONE_WARMUP_STEP_REDUCTION_FACTOR * nbr_iterations_max_phase_1
            eps_conv_phase_1 = OT_WO_Two_Phase.PHASE_ONE_WARMUP_CONVERGENCE_RELAX_FACTOR * eps_conv_phase_1

        (np_trans_weights, np_error_p1) = self._run_training(train_step=train_step_phase_1, 
                                                                nbr_iterations_max=nbr_iterations_max_phase_1, 
                                                                eps_convergence=eps_conv_phase_1, 
                                                                nbr_iterations_min = nbr_iterations_min_phase_1)
        training_time_p1 = time.time() - start_time_p1
        # Phase two should use new weights for sampling,
        # copying the SPN is a bit difficult because the mappings use the transitions
        # Hack: Save weights, change them, change them back
        np_trans_weights_initial = self._spn_container.get_weights()
        self._spn_container.update_transition_weights(np_trans_weights)

        
        ### Phase 2
        np_error_p2 = None
        training_time_p2 = None
        if self._run_phase_two:
            self._logger.info("Running Phase 2 Optimization")
            start_time_p2 = time.time()
            data_phase_two = DataPhaseTwoFactory.create_data_phase_two(self._spn_container, self._act_id_conn, 
                                                                       self._stoch_lang_log, 
                                                                       self._data_phase_1,
                                                                       self._max_nbr_paths, 
                                                                       self._max_nbr_variants, 
                                                                       self._nbr_samples_phase_two,
                                                                       self._emsc_loss_type)

            train_step_phase_2 = self._create_training_flex(
                optimizer=optimizer,
                data_phase_2=data_phase_two)
            (np_trans_weights, np_error_p2) = self._run_training(train_step=train_step_phase_2, 
                                                                 nbr_iterations_min=nbr_iterations_min, 
                                                                 nbr_iterations_max=nbr_iterations_max, 
                                                                 eps_convergence=eps_convergence)
            training_time_p2 = time.time() - start_time_p2
            # Reset weights
            self._spn_container.update_transition_weights(np_trans_weights_initial)

        time_opt = time.time() - time_opt_start
        self._time_full += time_opt
        self._logger.info(f'Optimized weights in {time_opt}s for {self.configuration_description}')
        
        return OT_WO_Result(np_trans_weights, np_error_p1, training_time_p1, np_error_p2, training_time_p2, self._time_full)
        
    def _run_training(self, train_step: Callable[[float], float], nbr_iterations_min=50, nbr_iterations_max=3000, eps_convergence=0.0025) -> tuple[np.ndarray, np.ndarray]:
        # Checking convergence requires HISTORY of errors
        nbr_iterations_min = max(50, nbr_iterations_min)
        error_series = []
        converged = False
        step = 0
        while step < nbr_iterations_max and not converged:
            loss_value = train_step((step + 1) / nbr_iterations_max)
            if step % 300 == 0:
                self._logger.debug(
                        "Training loss at step %d: %.4f"
                        % (step, float(loss_value))
                )
            error_series.append(float(loss_value))
            step += 1
            # Convergence criterion:
            # Check every 100 iterations
            # No major change within the last 50 iterations
            if step % 100 == 0 and step >= nbr_iterations_min:
                error_sum = abs(sum(error_series[-50:-25])  - sum(error_series[-25:]))
                converged = error_sum < eps_convergence

        return (self._path_variant_model.transition_weights, np.array(error_series))

    ################################################################################
    # Training Regimes:
    # Path Sample | Log Sample | Treatment
    # Fixed       | Fix        | V1: Special because cost matrix and variant likelihood are fixed in loss (avoids tf -> numpy)                         
    # Fixed       | Flex       | V2: Cost matrix and variant likelihoods change 
    # Flex        | Fix        | V2: Cost matrix changes and variant likelihoods fixed (could distinguish but simpler to use V2 function)                       
    # Flex        | Flex       | V2: Cost matrix and variant likelihoods change 
    ################################################################################
    def _create_training_regime_fixed_paths_fixed_variants(self, optimizer, data_phase_one: DataPhaseOne) -> Callable[[float], float]:
        """Training regime for FIXED path sample and FIXED log sample.

        Args:
            optimizer (_type_): Tensorflow (keras) optimizer.
            cost_matrix (np.ndarray[np.float32]): Cost matrix (numpy!)
            variant_lh (np.ndarray[np.float32]): Log variant likelihood (numpy!)

        Returns:
            Callable[[float], float]: Training step (progress -> EMD loss)
        """
        loss_ot = OT_WO_Two_Phase._create_loss_function_numpy(data_phase_one.variant_lh, data_phase_one.cost_matrix)
        model = self._path_variant_model
        nbr_weights = len(self._spn_container.net.transitions)
        add_extra_losses = self._path_variant_model.adds_extra_losses
        clip_gradients = self._clip_gradients

        #l_gradients = list()
        def train_step(progress: float):
            with tf.GradientTape() as tape:
                variant_probabilities = model.call((data_phase_one.tf_variant_2_paths, 
                                                    data_phase_one.tf_paths_nom, 
                                                    data_phase_one.tf_paths_denom))
                loss = loss_ot(variant_probabilities)
                if add_extra_losses:
                    loss += tf.add_n(model.losses)

            gradients = tape.gradient(loss, model.trainable_variables)
            if clip_gradients:
                gradients = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in gradients]
            #print(model.trainable_variables[0])
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            #l_gradients.append(tf.convert_to_tensor(gradients[0]))
            if progress > 0.99:
                pass
                #grad_sum = sum(l_gradients)
                #grad_abs_sum = sum([np.abs(g) for g in l_gradients])
                #print("Sum gradients:")
                #print(grad_sum)
                #print("Sum abs gradients:")
                #print(grad_abs_sum)
                #print("Back and forth gradients:")
                #print(np.divide(grad_sum, grad_abs_sum))
            return loss

        return train_step

    
    def _create_training_flex(self, optimizer, data_phase_2: DataPhaseTwo):
        """Training regime if model or log side are flexible. 
        In loss function, we will always have to convert cost matrix from tensorflow domain to numpy, which is costly.

        Setting only valid for PHASE TWO of optimization.
        Args:
            optimizer (_type_): Tensorflow (keras) optimizer
            data_phase_2 (DataPhaseTwo): Data container for phase 2.

        Returns:
            Callable[[float], float]: Training step (progress -> EMD loss)
        """
        # Closure
        model = self._path_variant_model
        add_extra_losses = self._path_variant_model.adds_extra_losses
        clip_gradients = self._clip_gradients

        # Actual train step
        def train_step(progress: float):
            data_iteration = data_phase_2.get_next_iteration_data()
            loss_emd, grads = OT_WO_Two_Phase._training_step_variable_ot_input(model, 
                    (data_iteration.tf_variant_2_paths, data_iteration.tf_paths_nom, data_iteration.tf_paths_denom),
                    data_iteration.variant_lh, data_iteration.cost_matrix, add_extra_losses)
            
            if clip_gradients:
                grads = [(tf.clip_by_value(grad, clip_value_min=-1.0, clip_value_max=1.0)) for grad in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return loss_emd

        return train_step

        
    ################################################################################
    # Loss Related Functionality
    ################################################################################
    def _training_step_variable_ot_input(model, inputs_model, b, C, add_extra_losses):

        @tf.custom_gradient
        def emd_variant_likelihood_loss(a):
            np_a = a.numpy()
            np_b = b.numpy()
            np_C = C.numpy()
            #if np.min(np_a) < 0.000001:
            #    print(np.min(np_a))
            try:
                (gamma, log) = ot.emd(np_a, np_b, np_C, log=True)
                direct_ot_success = True
            except AssertionError as error:
                module_logger.warning(f"Attempting to compute re-normalized ot because direct OT failed: {str(error)}")
                direct_ot_success = False
            
            if not direct_ot_success:
                np_a = np_a / np.sum(np_a)
                np_b = np_b / np.sum(np_b)
                (gamma, log) = ot.emd(np_a, np_b, np_C, log=True)

            def grad(dy):
                return dy * log['u']
            return log['cost'], grad


        with tf.GradientTape() as tape:
            variant_probabilities = model(inputs_model)
            emd_loss = emd_variant_likelihood_loss(variant_probabilities)
            if add_extra_losses:
                loss += tf.add_n(model.losses)

        return emd_loss, tape.gradient(emd_loss, model.trainable_variables)

    
    def _create_loss_function_numpy(b, M):
        @tf.custom_gradient
        def emd_variant_likelihood_loss(a):
            np_a = a.numpy()
            #if np.min(np_a) < 0.000001:
            #    print(np.min(np_a))
            try:
                (gamma, log) = ot.emd(np_a, b, M, log=True)
                direct_ot_success = True
            except AssertionError as error:
                module_logger.warning(f"Attempting to compute re-normalized ot because direct OT failed: {str(error)}")
                direct_ot_success = False
            
            if not direct_ot_success:
                np_a = np_a / np.sum(np_a)
                (gamma, log) = ot.emd(np_a, b, M, log=True)

            def grad(dy):
                return dy * log['u']
            return log['cost'], grad

        return emd_variant_likelihood_loss  

    
    @property
    def transition_weights(self):
        return self._path_variant_model.transition_weights

    @property
    def run_phase_two(self):
        # Final value is assigned in constructor
        return self._run_phase_two

    @property
    def data_phase_1(self):
        return self._data_phase_1

    @property
    def configuration_description(self) -> str:
        config_description = f'WaWE(loss={str(self._emsc_loss_type)}, layer={str(self._layer_type)}, '\
            f'#paths={self._max_nbr_paths}, #variants={self._max_nbr_variants}, '\
                f'2-phase={self._run_phase_two_init}->{self._run_phase_two} ({self._nbr_samples_phase_two} samples), ' \
                    f'hot-start={self._hot_start}, grad_clip={self._clip_gradients})'
        return config_description


            
