import logging.config
import logging
from multiprocessing import Pool
import multiprocessing
import traceback
import pm4py
from pathlib import Path
import argparse
import yaml
from ast import literal_eval
from typing import Iterable

from ot_backprop_pnwo.evaluation.evaluation_param import ConvergenceConfig
from ot_backprop_pnwo.evaluation.evaluation_reporting import EvaluationReporterTwoPhase, ExistingInitialRunRepo
from ot_backprop_pnwo.evaluation.evaluation_util import exec_log_model_evaluation_two_phase
from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType
from ot_backprop_pnwo.optimization.model import ResidualHandling
from ot_backprop_pnwo.spn import spn_util

TEST_MODE = False

logger = logging.getLogger(__name__)

def argparse_type_list_log_path_pn_directory(s: str):
    """ Argparse type: List of (log name, log path, Petri net (SPN) directory or file, warm start (True, False)) tuples

    The string passed should !evaluate! to an iterable (e.g., list) of tuples having the structure described above.

    Args:
        s (str): Python evaluatable string 

    Raises:
        argparse.ArgumentTypeError: If th string cannot be evaluated of the evaluate string does not contain the data

    Returns:
        _type_: _description_
    """
    type_hint = "Iterable[(log name: str, Log path: str, SPN directory/file:str , warm start[T/F]:bool)]"
    try:
        log_pn_input = literal_eval(s)
    except:
        print(s)
        raise argparse.ArgumentTypeError(f"Cannot Evaluate (ast.literal_eval): {type_hint}")
    # Check enumeration of tuples
    if not isinstance(log_pn_input, Iterable):
        raise argparse.ArgumentTypeError(f"Not an enumeration! {type_hint} should be a enumeration of triples.")
    if not all(isinstance(item, tuple) for item in log_pn_input):
        raise argparse.ArgumentTypeError(f"Not an enumeration of tuples! {type_hint} should be a enumeration of triples.")
    if not all(len(item) == 4 for item in log_pn_input):
        raise argparse.ArgumentTypeError(f"Not an enumeration of triples! {type_hint} should be a enumeration of triples.")
    if not all(isinstance(log_name, str) and isinstance(log_path, str) and isinstance(spn_path, str) and isinstance(warm_start, bool) for  (log_name, log_path, spn_path, warm_start) in log_pn_input):
        raise argparse.ArgumentTypeError(f"Tuple types (str, str, bool) incorrect ({type_hint}).")
    
    log_pn_as_paths = tuple((log_name, Path(log_path_str), Path(spn_path), warm_start) for (log_name, log_path_str, spn_path, warm_start) in log_pn_input)
    return log_pn_as_paths


def ot_sizes_type(s: str):
    """ Argparse type: Size tuple: 100,100
    Each tuple must be comma separated and must not contain whitespace characters.
    The entire list of entries is split by ; or whitespace

    Args:
        s (str): Argument string passed by argparse

    Raises:
        argparse.ArgumentTypeError: If parsing fails

    Returns:
        _type_: list of (size) pairs
    """
    try:
        ot_size_pair = tuple(map(int, s.split(',')))
        if len(ot_size_pair) != 2:
            argparse.ArgumentTypeError("OT-size entries need to be a white-space or semicolon sparated list of a comma-separated !pairs! of values")
        return ot_size_pair
    except:
        raise argparse.ArgumentTypeError("OT-size entries need to be a white-space or semicolon sparated list of a comma-separated pairs of values")

def main(path_tasks: Path, path_results: Path, ot_sizes: list, l_log_pn_pairs: list, conv_config: ConvergenceConfig, repetitions: int=20, 
         pool_size:int =10, emsc_loss_type:EMSCLossType=EMSCLossType.PEMSC, enable_phase_two=True, residual_handling: ResidualHandling=ResidualHandling.ADD_RESIDUAL_ELEMENT):
    # Init Existing Run Repository
    log_names_unique = set(log_name for (log_name, _, _, _) in l_log_pn_pairs)
    log_result_folder_paths = (path_results.joinpath(log_name) for log_name in log_names_unique)
    initially_exiting_runs = ExistingInitialRunRepo(log_result_folder_paths)
    first_free_id = initially_exiting_runs.get_next_free_id()

    with multiprocessing.Manager() as m:
        next_free_id = m.Value('runCounter', first_free_id)
        lock = m.Lock()
        with Pool(pool_size) as pool:
            for (log_name, log_path, pn_path, hot_start) in l_log_pn_pairs:
                logger.info(f'Start evaluation for {log_name} and using hot-start: {hot_start} models ({str(emsc_loss_type)}, {str(residual_handling)})')
                # Load log
                logger.info(f"Importing log {log_name}")
                log_path = path_tasks.joinpath(log_path)
                log = pm4py.read_xes(str(log_path))
                df_ev = pm4py.convert_to_dataframe(log)

                # Create result directory for log
                path_log_results = path_results.joinpath(log_name)
                path_log_results.mkdir(exist_ok=True)

                log_evaluation_reporter = EvaluationReporterTwoPhase(initially_exiting_runs, next_free_id, path_log_results, lock=lock)

                # Iterate over Petri nets
                l_eval_instances_log = None


                # Path Petri nets
                input_pn_path = path_tasks.joinpath(pn_path)
                if input_pn_path.is_dir():
                    pn_paths = input_pn_path.glob('*.pnml')
                else:
                    pn_paths = (input_pn_path, )

                for counter_pn, path_pn in enumerate(pn_paths):
                    net, im, fm = spn_util.load_as_spn(path_pn)
                    # Zero weights can already cause problems when sampling paths (before optimization even starts)
                    spn_util.ensure_non_zero_transition_weights(net)

                    name_spn = path_pn.name.replace('.pnml', '')
                    l_instances_log_pn = exec_log_model_evaluation_two_phase(evaluation_reporter=log_evaluation_reporter, 
                                                        df_ev=df_ev, net=net, im=im, fm=fm, name_log=log_name, name_spn=name_spn, 
                                                        ot_sizes=ot_sizes, hot_start=hot_start, iterations=repetitions, TEST_MODE=TEST_MODE, worker_pool=pool,
                                                        only_phase_one=not enable_phase_two,
                                                        print_missing_only=False,
                                                        convergence_configs=(conv_config, ),
                                                        emsc_loss_type=emsc_loss_type, residual_handling=residual_handling)
                    if l_eval_instances_log is None:
                        l_eval_instances_log = l_instances_log_pn
                    else:
                        l_eval_instances_log = l_eval_instances_log + l_instances_log_pn
                    if TEST_MODE and counter_pn >= 2:
                        break
        
                logger.info(f"Scheduled runs {len(l_eval_instances_log)} for {log_name}")
                logger.info("Waiting for termination before loading next log")

                if  l_eval_instances_log is not None:
                    for (param, future) in l_eval_instances_log:
                        try:
                            future.get()
                        except Exception as e:
                            logger.error(f'Evaluation run {str(EvaluationReporterTwoPhase._get_flattened_key_param_dict(param))} failed')
                            print(traceback.format_exc())
                            logger.error(str(e))

                logger.info(f"Executed {len(l_eval_instances_log)} evaluation runs")


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
        prog='Evaluation Loop Two-Phase OT Weight Optimization',
        description='Evaluates the two-phase SPN weight optimization method using Optimal Transport Theory'
    )
    # Positional
    parser.add_argument('taskDirectory', type=str)
    parser.add_argument('resultDirectory', type=str)
    parser.add_argument('emscLossType', type=EMSCLossType, choices=list(EMSCLossType))
    parser.add_argument('--residualHandling', type=ResidualHandling, choices=list(ResidualHandling), 
                        default=ResidualHandling.ADD_RESIDUAL_ELEMENT, help="How residual probability is handled (i.e., by adding a residual model or log trace or by normalization")
    parser.add_argument('--otSizes', help='White space/semicolon separated list of comma-separated size tuples (spn times logs)', 
                        type=ot_sizes_type, nargs='+',
                        default=[(400, 400), (400, 2000), (800, 800), (800, 2000)])
    parser.add_argument('--logNetData', help='Python evaluatable (ast.literal_eval) Iterable of (log path, Petri net (SPN) directory, warm start[T/F]) tuples', 
                        type=argparse_type_list_log_path_pn_directory, 
                        default= ( ('RTFM', 'logs/road_fines/road_fines.xes', 'models/road_fines/hot-start', True), 
                                  ('RTFM', 'logs/road_fines/road_fines.xes', 'models/road_fines/cold-start', False), 
                                  ('Sepsis', 'logs/sepsis/sepsis.xes', 'models/sepsis/hot-start', True), 
                                  ('Sepsis', 'logs/sepsis/sepsis.xes', 'models/sepsis/cold-start', False), 
                                  ('BPIC18Ref', 'logs/bpic-2018-reference/bpic-2018-reference.xes', 'models/bpic-2018-reference/hot-start', True), 
                                  ('BPIC18Ref', 'logs/bpic-2018-reference/bpic-2018-reference.xes', 'models/bpic-2018-reference/cold-start', False),
                                )
                        )
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--poolSize', type=int, default=10)
    # Convergence Criterion
    parser.add_argument('--convMinIter', type=int, default=50)
    parser.add_argument('--convMaxIter', type=int, default=5000)
    parser.add_argument('--convEps', type=float, default=0.0025)
    # Phase II
    parser.add_argument('--enablePhaseII', action='store_true', help="Enables Phase II (if enabled, evaluation will run both (Phase I only, and Phase I + II if required)")

    args = parser.parse_args()
    path_tasks = Path(args.taskDirectory)
    path_results = Path(args.resultDirectory)
    repetitions = args.repetitions
    emsc_loss_type = args.emscLossType
    residual_handling = args.residualHandling
    ot_sizes = args.otSizes
    pool_size = args.poolSize
    l_log_pn_pairs = args.logNetData
    phase_two_enabled = args.enablePhaseII
    conv_config = ConvergenceConfig(args.convMinIter, args.convMaxIter, args.convEps)

    main(path_tasks, path_results, ot_sizes, l_log_pn_pairs, conv_config, 
         repetitions=repetitions, pool_size=pool_size, emsc_loss_type=emsc_loss_type, enable_phase_two=phase_two_enabled, residual_handling=residual_handling)
