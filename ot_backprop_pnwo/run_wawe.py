import dataclasses
import logging.config
import logging
from multiprocessing import Pool
import multiprocessing
import numpy as np
import pm4py
from pathlib import Path
import argparse
import tensorflow as tf
import yaml

from ot_backprop_pnwo.evaluation.evaluation_param import ConvergenceConfig
from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType
from ot_backprop_pnwo.optimization.model import Path2VariantLayerTypes
from ot_backprop_pnwo.optimization.model import ResidualHandling
from ot_backprop_pnwo.optimization.ot_wo_two_phase import OT_WO_Two_Phase
from ot_backprop_pnwo.spn import spn_util
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang

TEST_MODE = False

logger = logging.getLogger(__name__)

def main(path_log: Path, path_pn: Path, path_spn_out: Path, emsc_loss_type: EMSCLossType, 
         max_nbr_paths: int, max_nbr_variants: int,
         layer_type: Path2VariantLayerTypes,
         residual_handling: ResidualHandling,
         conv_config: ConvergenceConfig, warm_start: bool, phase_two_enabled: bool): 
    ########## Load and Prepare Data ##########
    # Log
    logger.info(f"Importing log {str(path_log)}")
    log = pm4py.read_xes(str(path_log))
    df_ev = pm4py.convert_to_dataframe(log)
    # Net
    logger.info("Loading Net")
    net, im, fm = spn_util.load_as_spn(path_pn)
    # Zero weights can already cause problems when sampling paths (before optimization even starts)
    spn_util.ensure_non_zero_transition_weights(net)
    # Connect
    act_id_conn = ActivityIDConnector(df_ev, net)
    spn_container = SPNWrapper(act_id_conn, net, im, fm)
    stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)

    ########## Optimization ##########
    logger.info("Running weight estimation using subgradient method")
    spn_wo_alg = OT_WO_Two_Phase(spn_container=spn_container, stoch_lang_log=stoch_lang_log, 
                                 act_id_conn=act_id_conn,
                                 hot_start=warm_start, run_phase_two=phase_two_enabled,
                                 max_nbr_paths=max_nbr_paths, max_nbr_variants=max_nbr_variants,
                                 layer_type=layer_type,
                                 emsc_loss_type=emsc_loss_type,
                                 residual_handling=residual_handling)

    optimizer = tf.keras.optimizers.Adam(0.001)
    ot_wo_result = spn_wo_alg.optimize_weights(optimizer, **dataclasses.asdict(conv_config))
    str_loss_reduction = f"Loss reduction: {ot_wo_result.error_series_phase_one[0]} (initial Phase I)"
    str_loss_reduction += f" -> {ot_wo_result.error_series_phase_one[-1]} (Phase I final iteration {len(ot_wo_result.error_series_phase_one)})"
    if ot_wo_result.error_series_phase_two is not None:
        str_loss_reduction += f" -> {ot_wo_result.error_series_phase_two[0]} (Phase II initial)"
        str_loss_reduction += f" -> {ot_wo_result.error_series_phase_two[-1]} (Phase II final iteration {len(ot_wo_result.error_series_phase_two)})"
    logger.info(str_loss_reduction)
    logger.info(f"Completed weight estimation in {ot_wo_result.full_time}s")
    ########## Write Results ##########
    logger.info("Exporting results")
    spn_container.update_transition_weights(np.abs(ot_wo_result.spn_weights))
    spn_container.export_to_file(path_spn_out)
    logger.info("Exported optimized SPN")


if __name__ == '__main__':
    # Load the config file
    with open('ot_backprop_pnwo/logger-config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    print(config)
    # Configure the logging module with the config file
    logging.config.dictConfig(config)

    ##############################
    # Arguments
    ##############################
    parser = argparse.ArgumentParser(
        prog='Run WAsserstein Weight Estimation (WAWE) for Stochastic Petri Nets',
        description='Optimize the weights of a given SPN (Petri net) to maximize EMSC.'
    )
    # Positional
    parser.add_argument('pathLog', type=str, help="Path to the event log (XES)")
    parser.add_argument('pathPN', type=str, help="Path to the SPN or Petri net")
    parser.add_argument('pathOutput', type=str, help="Output path to which resulting SPN is written (folder must exist)")
    # Configurations
    parser.add_argument('maxNbrSPNPaths', type=int, help="Maximum nbr of paths sampled from the net")
    parser.add_argument('maxNbrVariants', type=int, help="Maximum nbr of trace variants sampled from the event log's stochastic language")
    parser.add_argument('--emscLossType', type=EMSCLossType, choices=list(EMSCLossType), 
                        default=EMSCLossType.PEMSC, help="OT-loss formulation used for optimization")
    parser.add_argument('--residualHandling', type=ResidualHandling, choices=list(ResidualHandling), 
                        default=ResidualHandling.ADD_RESIDUAL_ELEMENT, help="How residual probability is handled (i.e., by adding a residual model or log trace or by normalization")
    # Convergence Criterion
    parser.add_argument('--convMinIter', type=int, default=50)
    parser.add_argument('--convMaxIter', type=int, default=5000)
    parser.add_argument('--convEps', type=float, default=0.0025)
    # Phase II
    parser.add_argument('--enablePhaseII', action='store_true', help="Enables Phase II (if enabled, evaluation will run both (Phase I only, and Phase I + II if required)")
    parser.add_argument('--warmStart', action='store_true', help="Perform a warm start (requires that input net is an SPN)")

    args = parser.parse_args()
    path_log = Path(args.pathLog)
    path_pn = Path(args.pathPN)
    path_spn_out = Path(args.pathOutput)
    emsc_loss_type = args.emscLossType
    residual_handling = args.residual_handling
    max_nbr_paths = args.maxNbrSPNPaths
    max_nbr_variants = args.maxNbrVariants
    conv_config = ConvergenceConfig(args.convMinIter, args.convMaxIter, args.convEps)
    phase_two_enabled = args.enablePhaseII
    warm_start = args.warmStart

    main(path_log, path_pn, path_spn_out, emsc_loss_type, 
         max_nbr_paths, max_nbr_variants, 
         Path2VariantLayerTypes.EXP_LOG_ABS,
         conv_config, warm_start, phase_two_enabled, residual_handling) 
