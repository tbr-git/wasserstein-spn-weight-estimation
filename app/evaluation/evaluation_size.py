import argparse
import logging
import numpy as np
from pathlib import Path
import pm4py

from ot_backprop_pnwo.evaluation.evaluation_reporting import EvaluationReporterTwoPhase
from ot_backprop_pnwo.evaluation.evaluation_util import exec_log_model_evaluation_two_phase
from ot_backprop_pnwo.spn import spn_util


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ##############################
    # Arguments
    ##############################
    parser = argparse.ArgumentParser(
        prog='Evaluation Loop OT Weight Optimization',
        description='Evaluates'
    )
    parser.add_argument('taskDirectory', type=str)
    parser.add_argument('resultDirectory', type=str)
    args = parser.parse_args()
    path_tasks = Path(args.taskDirectory)
    path_results = Path(args.resultDirectory)

    # Log - Petri net folder
    l_log_pn_pairs = (
        ('RTFM', 'logs/road_fines/road_fines.xes', 'models/road_fines/cold-start/rtfm_IMf02_ABE.pnml'),
        ('Sepsis', 'logs/sepsis/sepsis.xes', 'models/sepsis/cold-start'),
    )
    ot_sizes = ((200, 200), (400, 400), (600, 600), (800, 800), (1_000, 1_000), (3_000, 3_000), (5_000, 5_000))
    iterations = 10

    for (log_name, log_path, pn_path) in l_log_pn_pairs:
        # Load log
        logging.info("Importing log")
        log_path = path_tasks.joinpath(log_path)
        print(log_path)
        log = pm4py.read_xes(str(log_path))
        df_ev = pm4py.convert_to_dataframe(log)
        # Create result directory for log
        path_log_results = path_results.joinpath(log_name)
        path_log_results.mkdir(exist_ok=True)
        path_result_info_file = path_log_results / 'run-info.json'

        # Path Petri nets
        input_pn_path = path_tasks.joinpath(pn_path)
        if input_pn_path.is_dir():
            pn_paths = input_pn_path.glob('*.pnml')
        else:
            pn_paths = (input_pn_path, )


        log_evaluation_reporter = EvaluationReporterTwoPhase(path_log_results, path_result_info_file, override=True)

        # Iterate over Petri nets
        for path_pn in pn_paths:
            net, im, fm = spn_util.load_as_spn(path_pn)
            # Zero weights can already cause problems when sampling paths (before optimization even starts)
            spn_util.ensure_non_zero_transition_weights(net)

            name_spn = path_pn.name.replace('.pnml', '')
            logging.info(f"Evaluating {name_spn} on log {log_name}...")

            exec_log_model_evaluation_two_phase(evaluation_reporter=log_evaluation_reporter, 
                                                df_ev=df_ev, net=net, im=im, fm=fm, name_log=log_name, name_spn=name_spn, 
                                                convergence_configs=None,
                                                ot_sizes=ot_sizes, hot_start=False, iterations=iterations)
