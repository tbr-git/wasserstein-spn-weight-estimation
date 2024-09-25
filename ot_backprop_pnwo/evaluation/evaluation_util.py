import dataclasses
import inspect
from typing import Optional
import tensorflow as tf
import logging
from ot_backprop_pnwo.evaluation.evaluation_param import ConvergenceConfig, TwoPhaseRunUniqueIdentifyingConfig
from ot_backprop_pnwo.evaluation.evaluation_reporting import EvaluationReporterTwoPhase
from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType
from ot_backprop_pnwo.optimization.model import Path2VariantLayerTypes, ResidualHandling

from ot_backprop_pnwo.optimization.ot_wo_two_phase import OT_WO_Two_Phase
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang

logger = logging.getLogger(__name__)


def exec_log_model_evaluation_two_phase(evaluation_reporter: EvaluationReporterTwoPhase,
                                        df_ev, net, im, fm, name_log: str, name_spn: str, 
                                        ot_sizes: tuple[tuple[int, int]], 
                                        hot_start:bool,
                                        iterations: int, 
                                        only_phase_one: bool=False,
                                        convergence_configs: Optional[tuple[ConvergenceConfig]]=None,
                                        TEST_MODE: bool=False,
                                        worker_pool=None,
                                        print_missing_only=False,
                                        emsc_loss_type=EMSCLossType.PEMSC,
                                        residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT
                            ):
    act_id_conn = ActivityIDConnector(df_ev, net)
    spn_container = SPNWrapper(act_id_conn, net, im, fm)
    stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)
    if TEST_MODE:
        iterations = 2

    l_running_instances = list()
    if convergence_configs is None:
        convergence_configs = (None, )

    for (max_nbr_paths, max_nbr_variants) in ot_sizes:
        for conv_config in convergence_configs:
            if conv_config is None:
                conv_config = ConvergenceConfig(**get_default_args(OT_WO_Two_Phase.optimize_weights))
            
            
            for i in range(iterations):
                key_parameterization = TwoPhaseRunUniqueIdentifyingConfig(
                    emsc_loss_type = emsc_loss_type,
                    name_log = name_log,
                    name_spn = name_spn,
                    max_nbr_paths = max_nbr_paths,
                    max_nbr_variants = max_nbr_variants,
                    hot_start = hot_start,
                    iteration = i,
                    phase_two_enabled = not only_phase_one,
                    convergence_config= conv_config,
                    residual_handling = residual_handling
                )
                if worker_pool is None:
                    _exec_evaluation_both_phases(evaluation_reporter=evaluation_reporter, key_parameterization=key_parameterization, act_id_conn=act_id_conn, spn_container=spn_container, 
                                                stoch_lang_log=stoch_lang_log, only_phase_one=only_phase_one, TEST_MODE=TEST_MODE, print_missing_only=print_missing_only)
                else:
                    l_running_instances.append(
                        (key_parameterization, 
                         worker_pool.apply_async(_exec_evaluation_both_phases, 
                                                 args = (evaluation_reporter, key_parameterization, act_id_conn, spn_container, stoch_lang_log, only_phase_one, TEST_MODE, print_missing_only))
                        ))
    return l_running_instances


def _exec_evaluation_both_phases(evaluation_reporter: EvaluationReporterTwoPhase, key_parameterization: TwoPhaseRunUniqueIdentifyingConfig, 
                                act_id_conn: ActivityIDConnector,
                                spn_container: SPNWrapper,
                                stoch_lang_log: StochasticLang,
                                only_phase_one=False,
                                TEST_MODE=False,
                                print_missing_only=False
                                ):
    logger.info(f"Running Evaluation (Phase I, Phase II optionally) {str(EvaluationReporterTwoPhase._get_flattened_key_param_dict(key_parameterization))}")
    phase_two_executed = None
    if not only_phase_one:
        ##############################
        # Run Two-Phase Enabled
        ##############################
        key_param = dataclasses.replace(key_parameterization, phase_two_enabled=True)
        if evaluation_reporter.is_run_required(key_param):
            if print_missing_only:
                logger.info(f"Would execute run: {key_param}")
            else:
                phase_two_executed = _exec_evaluation_two_phase_from_key_parameters(evaluation_reporter, 
                                                                            act_id_conn, spn_container, stoch_lang_log, 
                                                                            key_param, 
                                                                            TEST_MODE=TEST_MODE)
        else:
            logger.info(f"Skip Phase I")
            phase_two_executed = evaluation_reporter.is_phase_two_executed_in_run(key_param)

    ##############################
    # First Phase 
    # Explicitly run first phase test, if second phase did not fall back to first phase only
    # (due to the model's and log's language being small)
    ##############################
    #if False and (only_phase_one or phase_two_executed is None or phase_two_executed):
    if only_phase_one or phase_two_executed is None or phase_two_executed:
        key_param = dataclasses.replace(key_parameterization, phase_two_enabled=False)
        if evaluation_reporter.is_run_required(key_param):
            if print_missing_only:
                logger.info(f"Would execute run: {key_param}")
            else:
                phase_two_executed = _exec_evaluation_two_phase_from_key_parameters(evaluation_reporter, 
                                                                            act_id_conn, spn_container, stoch_lang_log, 
                                                                            key_param, TEST_MODE=TEST_MODE)
        else:
            logger.debug(f"Skip Phase II")


def _exec_evaluation_two_phase_from_key_parameters(evaluation_reporter: EvaluationReporterTwoPhase,
                                                  act_id_conn: ActivityIDConnector, spn_container: SPNWrapper, stoch_lang_log: StochasticLang, 
                                                  key_param: TwoPhaseRunUniqueIdentifyingConfig,
                                                  TEST_MODE: bool=False):
    spn_wo_alg = OT_WO_Two_Phase(spn_container=spn_container, stoch_lang_log=stoch_lang_log, 
                                act_id_conn=act_id_conn,
                                hot_start=key_param.hot_start, run_phase_two=key_param.phase_two_enabled,
                                max_nbr_paths=key_param.max_nbr_paths, max_nbr_variants=key_param.max_nbr_variants,
                                layer_type=Path2VariantLayerTypes.EXP_LOG_ABS,
                                emsc_loss_type=key_param.emsc_loss_type,
                                residual_handling=key_param.residual_handling)

    optimizer = tf.keras.optimizers.Adam(0.001)
    if TEST_MODE:
        ot_wo_result = spn_wo_alg.optimize_weights(optimizer, nbr_iterations_max=500)
    else:
        ot_wo_result = spn_wo_alg.optimize_weights(optimizer, **dataclasses.asdict(key_param.convergence_config))

    evaluation_reporter.report(key_param, spn_container, ot_wo_result, spn_wo_alg)
    return spn_wo_alg.run_phase_two


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }   

