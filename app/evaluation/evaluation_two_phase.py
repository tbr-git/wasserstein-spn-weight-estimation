import logging.config
import logging
from multiprocessing import Pool, Value
import multiprocessing
import re
import pm4py
from pathlib import Path
import argparse

import yaml

from ot_backprop_pnwo.evaluation.evaluation_reporting import EvaluationReporterTwoPhase, ExistingInitialRunRepo
from ot_backprop_pnwo.evaluation.evaluation_util import exec_log_model_evaluation_two_phase
from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType
from ot_backprop_pnwo.spn import spn_util

TEST_MODE = False

logger = logging.getLogger(__name__)


def ot_sizes_type(s):
    seps_size_tuples = r'[ ;]'
    try:
        ot_size_pairs = []
        for si in re.split(seps_size_tuples, s):
            ot_size_pair = tuple(map(int, si.split(',')))
            if len(ot_size_pair) != 2:
                argparse.ArgumentTypeError("OT-size entries need to be a white-space or semicolon sparated list of a comma-separated !pairs! of values")
            ot_size_pairs.append(ot_size_pair)

        return ot_size_pairs
    except:
        raise argparse.ArgumentTypeError("OT-size entries need to be a white-space or semicolon sparated list of a comma-separated pairs of values")


def main(path_tasks: Path, path_results: Path, ot_sizes: list, l_log_pn_pairs: list, repetitions: int, pool_size=10, emsc_loss_type=EMSCLossType.PEMSC):
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
                logger.info(f'Start evaluation for {log_name} and using hot-start: {hot_start} models')
                # Load log
                logger.info(f"Importing log {log_name}")
                log_path = path_tasks.joinpath(log_path)
                print(log_path)
                log = pm4py.read_xes(str(log_path))
                df_ev = pm4py.convert_to_dataframe(log)

                # Path Petri nets
                dir_pn = path_tasks.joinpath(pn_path)
                # Create result directory for log
                path_log_results = path_results.joinpath(log_name)
                path_log_results.mkdir(exist_ok=True)

                log_evaluation_reporter = EvaluationReporterTwoPhase(initially_exiting_runs, next_free_id, path_log_results, lock=lock)

                # Iterate over Petri nets
                l_eval_instances_log = None
                for counter_pn, path_pn in enumerate(dir_pn.glob('*.pnml')):
                    net, im, fm = spn_util.load_as_spn(path_pn)
                    # Zero weights can already cause problems when sampling paths (before optimization even starts)
                    spn_util.ensure_non_zero_transition_weights(net)

                    name_spn = path_pn.name.replace('.pnml', '')
                    l_instances_log_pn = exec_log_model_evaluation_two_phase(evaluation_reporter=log_evaluation_reporter, 
                                                        df_ev=df_ev, net=net, im=im, fm=fm, name_log=log_name, name_spn=name_spn, 
                                                        ot_sizes=ot_sizes, hot_start=hot_start, iterations=repetitions, TEST_MODE=TEST_MODE, worker_pool=pool,
                                                        print_missing_only=False,
                                                        emsc_loss_type=emsc_loss_type)
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
                            print(e)

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
    parser.add_argument('taskDirectory', type=str)
    parser.add_argument('resultDirectory', type=str)
    parser.add_argument('emscLossType', type=EMSCLossType, choices=list(EMSCLossType))
    parser.add_argument('--otSizes', help='White space/semicolon separated list of comma-separated size tuples (spn times logs)', 
                        type=ot_sizes_type, nargs='+',
                        default=[(400, 400), (400, 2000), (800, 800), (800, 2000)])
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--poolSize', type=int, default=10)
    args = parser.parse_args()
    path_tasks = Path(args.taskDirectory)
    path_results = Path(args.resultDirectory)
    repetitions = args.repetitions
    emsc_loss_type = args.emscLossType
    ot_sizes = args.otSizes
    pool_size = args.poolSize

    # Log - Petri net folder
    l_log_pn_pairs = (
        ('RTFM', 'logs/road_fines/road_fines.xes', 'models/road_fines/hot-start', True),
        ('RTFM', 'logs/road_fines/road_fines.xes', 'models/road_fines/cold-start', False),
        ('Sepsis', 'logs/sepsis/sepsis.xes', 'models/sepsis/hot-start', True),
        ('Sepsis', 'logs/sepsis/sepsis.xes', 'models/sepsis/cold-start', False),
        ('BPIC18Ref', 'logs/bpic-2018-reference/bpic-2018-reference.xes', 'models/bpic-2018-reference/hot-start', True),
        ('BPIC18Ref', 'logs/bpic-2018-reference/bpic-2018-reference.xes', 'models/bpic-2018-reference/cold-start', False),
    )
    main(path_tasks, path_results, ot_sizes, l_log_pn_pairs, repetitions=repetitions, pool_size=pool_size, emsc_loss_type=emsc_loss_type)
