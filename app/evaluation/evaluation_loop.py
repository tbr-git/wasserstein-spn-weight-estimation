import tensorflow as tf
import numpy as np
import numpy.typing as npt
import logging
import pm4py
import pandas as pd
from pathlib import Path
import argparse

from ot_backprop_pnwo.optimization.spn_wo_method import SPNWOMethod
from ot_backprop_pnwo.spn import spn_util
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang



def _process_petri_net(df_ev, net, im):
    log_activities = df_ev['concept:name'].unique()

    # Consider all activities not contained in log silent
    # (Current PM4Py import makes ProM tau transitions visible)
    for t in net.transitions:
        if t.label is not None and not (t.label in log_activities):
            logging.warning(f"Consider {t.label} silent")
            t.label = None

    # No initial marking found
    # Set initial marking assuming that the Petri net
    # is a workflow net
    if len(im) == 0:
        logging.warning("Initial marking of SPN is empty. Creating one under workflow net assumption.")
        for p in net.places:
            # No incoming arcs
            if len(p.in_arcs) == 0:
                im[p] += 1
        logging.warning(f"Created initial marking {str(im)}")


def exec_log_model_evaluation(df_ev, net, im, fm, name_log: str, name_spn: str, 
                              path_results: Path, id_start: int, iterations: int=5):
    act_id_conn = ActivityIDConnector(df_ev, net)
    spn_container = SPNWrapper(act_id_conn, net, im, fm)
    stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)

    for ot_size in [100, 200, 300, 400]:
        # Size sufficient to capture complete log 
        size_unrolls_log = ot_size >= len(stoch_lang_log)
        # Size sufficient to fully unroll SPN
        size_unrolls_spn = spn_util.spn_allows_for_at_least_paths(spn_container, ot_size)

        ##############################
        # Fixed run
        ##############################
        nbr_fixed_runs = iterations
        if size_unrolls_log and size_unrolls_spn:
            nbr_fixed_runs = 1
        ### Hot Start
        logging.info(f"Evaluating fixed OT size {ot_size} - hot start")
        for i in range(nbr_fixed_runs):
            spn_wo_method = SPNWOMethod(spn_container, stoch_lang_log, act_id_conn, 
                                        ot_size, ot_size, 
                                        fix_paths=True, fix_log_lang=True,
                                        t_weight_hot_start=True)

            optimizer = tf.keras.optimizers.Adam(0.001)
            (updated_weights, error_series, training_time) = spn_wo_method.run_training_regime_simple(optimizer, 3000)
            spn_container.update_transition_weights(np.abs(updated_weights))
            _write_results(path_results, id_start, spn_container, error_series, name_log=name_log, name_spn=name_spn,
                           size_unrolls_log=size_unrolls_log, size_unrolls_spn=size_unrolls_spn, fix_log_lang=True, fix_paths=True, 
                           hot_start=True, ot_size=ot_size, iteration=i, training_time=training_time)
            id_start += 1

        ### No hot Start
        logging.info(f"Evaluating fixed OT size {ot_size} - no hot start")
        for i in range(nbr_fixed_runs):
            spn_wo_method = SPNWOMethod(spn_container, stoch_lang_log, act_id_conn, 
                                        ot_size, ot_size, 
                                        fix_paths=True, fix_log_lang=True,
                                        t_weight_hot_start=False)

            optimizer = tf.keras.optimizers.Adam(0.001)
            (updated_weights, error_series, training_time) = spn_wo_method.run_training_regime_simple(optimizer, 3000)
            spn_container.update_transition_weights(np.abs(updated_weights))
            _write_results(path_results, id_start, spn_container, error_series, name_log=name_log, name_spn=name_spn,
                           size_unrolls_log=size_unrolls_log, size_unrolls_spn=size_unrolls_spn, fix_log_lang=True, fix_paths=True, 
                           hot_start=False, ot_size=ot_size, iteration=i, training_time=training_time)
            id_start += 1

        ##############################
        # Flexible run (if makes sense)
        ##############################
        logging.info(f"Evaluating fixed OT size {ot_size} - no hot start")
        if not size_unrolls_log or not size_unrolls_spn:
            ### Hot Start
            logging.info(f"Evaluating flex OT size {ot_size} - hot start")
            for i in range(iterations):
                spn_wo_method = SPNWOMethod(spn_container, stoch_lang_log, act_id_conn, 
                                            max_nbr_paths=ot_size, max_nbr_variants=ot_size,
                                            fix_paths=None, fix_log_lang=None,
                                            t_weight_hot_start=True)

                optimizer = tf.keras.optimizers.Adam(0.001)
                (updated_weights, error_series, training_time) = spn_wo_method.run_training_regime_simple(optimizer, 3000, nbr_samples=5)
                spn_container.update_transition_weights(np.abs(updated_weights))

                _write_results(path_results, id_start, spn_container, error_series, name_log=name_log, name_spn=name_spn,
                            size_unrolls_log=size_unrolls_log, size_unrolls_spn=size_unrolls_spn, 
                            fix_log_lang=spn_wo_method.fix_log_lang, fix_paths=spn_wo_method.fix_paths, 
                            hot_start=True, ot_size=ot_size, iteration=i, training_time=training_time)
                id_start += 1

            ### No hot Start
            logging.info(f"Evaluating flex OT size {ot_size} - no hot start")
            for i in range(iterations):
                spn_wo_method = SPNWOMethod(spn_container, stoch_lang_log, act_id_conn, 
                                            max_nbr_paths=ot_size, max_nbr_variants=ot_size,
                                            fix_paths=None, fix_log_lang=None,
                                            t_weight_hot_start=False)

                optimizer = tf.keras.optimizers.Adam(0.001)
                (updated_weights, error_series, training_time) = spn_wo_method.run_training_regime_simple(optimizer, 3000, nbr_samples=5)
                spn_container.update_transition_weights(np.abs(updated_weights))

                _write_results(path_results, id_start, spn_container, error_series, name_log=name_log, name_spn=name_spn,
                            size_unrolls_log=size_unrolls_log, size_unrolls_spn=size_unrolls_spn, 
                            fix_log_lang=spn_wo_method.fix_log_lang, fix_paths=spn_wo_method.fix_paths, 
                            hot_start=False, ot_size=ot_size, iteration=i, training_time=training_time)
                id_start += 1

    return id_start


def exec_log_model_evaluation_test_barrier(df_ev, net, im, fm, name_log: str, name_spn: str, 
                              path_results: Path, id_start: int, iterations: int=10):
    act_id_conn = ActivityIDConnector(df_ev, net)
    spn_container = SPNWrapper(act_id_conn, net, im, fm)
    stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)

    # For a first test, I only consider size 400
    for i in range(iterations):
        # Weight Clipping
        spn_wo_method = SPNWOMethod(spn_container, stoch_lang_log, act_id_conn, 
                                    400, 400, 
                                    fix_paths=True, fix_log_lang=True,
                                    t_weight_hot_start=False,
                                    t_weights_log_barrier=False, 
                                    t_weights_regularize=False,
                                    t_weights_clip=False)

        # Log barrier and regularization
        #nbr_transitions = len(spn_container.net.transitions)
        #spn_wo_method = SPNWOMethod(spn_container, stoch_lang_log, act_id_conn, 
        #                        400, 400, 
        #                        fix_paths=True, fix_log_lang=True,
        #                        t_weight_hot_start=False,
        #                        t_weights_log_barrier=True, t_weights_barrier_mu=0.01,
        #                        t_weights_regularize=True, t_weights_reg_f=0.01 / nbr_transitions,
        #                        t_weights_clip=False)

        optimizer = tf.keras.optimizers.Adam(0.001)
        #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        (updated_weights, error_series, training_time) = spn_wo_method.run_training_regime_simple(optimizer, 4000)
        spn_container.update_transition_weights(np.abs(updated_weights))
        _write_results(path_results, id_start, spn_container, error_series, name_log=name_log, name_spn=name_spn,
                        size_unrolls_log=None, size_unrolls_spn=None, fix_log_lang=True, fix_paths=True, 
                        hot_start=False, ot_size=400, iteration=i, training_time=training_time)
        id_start += 1

    return id_start


def _write_results(path_results:Path, id_run: int, spn_container: SPNWrapper, error_series: npt.NDArray[np.float_], name_log: str, name_spn: str, 
                   size_unrolls_log: bool, size_unrolls_spn: bool, fix_log_lang: bool, fix_paths: bool, hot_start: bool,
                   ot_size: int, iteration: int, training_time: int):
    path_spn = path_results.joinpath(f"{name_spn}-{id_run}.pnml" )
    spn_container.export_to_file(path_spn)
    path_error = path_results.joinpath(f"error-series-{id_run}.csv")
    np.savetxt(path_error, error_series, delimiter=",")
    path_desc = path_results.joinpath("descriptions.csv")
    with open(path_desc, "a") as f:
        f.write(f"{id_run},{name_log},{name_spn},{size_unrolls_log},{size_unrolls_spn},{fix_log_lang},{fix_paths},{hot_start},{ot_size},{iteration},{training_time}\n")

def _write_header(path_results:Path):
    path_desc = path_results.joinpath("descriptions.csv")
    with open(path_desc, "a") as f:
        f.write(f"IdRun,NameLog,NameSPN,CompleteLog,CompletePaths,FixLogLang,FixSPNPaths,HotStart,OTSize,Iteration,TrainingTime\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
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
    l_log_pn_pairs = (('RTFM', 'logs/road_fines/road_fines.xes', 'models-cold-start/road_fines'), )

    for (log_name, log_path, pn_path) in l_log_pn_pairs:
        # Load log
        logging.info("Importing log")
        log_path = path_tasks.joinpath(log_path)
        print(log_path)
        #log = pm4py.read_xes(str(log_path))
        df_ev = pd.read_csv('./RTFM_as_CSV.csv', sep=',')
        df_ev = pm4py.format_dataframe(df_ev, case_id='case', activity_key='event', timestamp_key='completeTime')
        #df_ev = pm4py.convert_to_dataframe(log)
        # Path Petri nets
        dir_pn = path_tasks.joinpath(pn_path)
        id_start = 0
        # Create result directory for log
        path_log_results = path_results.joinpath(log_name)
        path_log_results.mkdir(exist_ok=True)

        _write_header(path_log_results)
        # Iterate over Petri nets
        for path_pn in dir_pn.glob('*.pnml'):
            net, im, fm = spn_util.load_as_spn(path_pn)
            # After bugfix in pm4py, not needed anymore
            # _process_petri_net(df_ev, net, im)
            name_spn = path_pn.name.replace('.pnml', '')
            logging.info(f"Evaluating {name_spn} on log {log_name}...")
            id_start = exec_log_model_evaluation_test_barrier(df_ev, net, im, fm, log_name, name_spn, path_log_results, id_start)



