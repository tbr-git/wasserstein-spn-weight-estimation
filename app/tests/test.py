import numpy as np
import logging
from ot_backprop_pnwo.optimization.model import Path2VariantModelFactory
from ot_backprop_pnwo.optimization.spn_wo_method import SPNWOMethod
import tensorflow as tf
from ot_backprop_pnwo.spn import spn_util

from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang
from ..evaluation.evaluation_loop import _process_petri_net

import pm4py
import pandas as pd
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    ### Event Log
    logging.info("Importing log")
    df_ev = pd.read_csv('./RTFM_as_CSV.csv', sep=',')
    df_ev = pm4py.format_dataframe(df_ev, case_id='case', activity_key='event', timestamp_key='completeTime')

    ##### Load Net and Event Log Side #####
    # SPN
    #net, im, fm, stochastic_map = pnml_importer.apply("spn-by-hm.pnml", parameters={"return_stochastic_map": True})
    net, im, fm = spn_util.load_as_spn("./ot-wo-evaluation/data/models/road_fines/cold-start/rtfm_RDS_globalPreMurata.pnml")
    #net, im, fm, stochastic_map = pnml_importer.apply("./ot-wo-evaluation/data/models-cold-start/road_fines/road_fines_IMf02_ABE.pnml", parameters={"return_stochastic_map": True})

    act_id_conn = ActivityIDConnector(df_ev, net)
    spn_container = SPNWrapper(act_id_conn, net, im, fm)
    stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)

    nbr_transitions = len(spn_container.net.transitions)

    # Normal domain
    # For a first test, I only consider size 400
    fn_model_creation = lambda add_res_path: Path2VariantModelFactory.init_base_model_abs(nbr_transitions=nbr_transitions, add_res_prob_path=add_res_path)

    spn_wo_method = SPNWOMethod(spn_container, stoch_lang_log, act_id_conn, 
        fn_factory_path_variant_model=fn_model_creation,
        max_nbr_paths=400, max_nbr_variants=400,
        fix_paths=False, fix_log_lang=False)

    print('Initial weights normal domain')
    print(spn_wo_method.transition_weights)

    optimizer = tf.keras.optimizers.Adam(0.001)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    (updated_weights, error_series, training_time) = spn_wo_method.run_training_regime_simple(optimizer, 3000)
    print(updated_weights)
    spn_container.update_transition_weights(updated_weights)
    spn_container.export_to_file('rtfm_RDS_globalPreMurata-optimized.pnml')

    exit(0)
    # Log domain
    # For a first test, I only consider size 400

    fn_model_creation_log = lambda add_res_path: Path2VariantModelFactory.init_log_model_abs(nbr_transitions=nbr_transitions, add_res_prob_path=add_res_path)

    spn_wo_method_log = SPNWOMethod(spn_container, stoch_lang_log, act_id_conn, 
        fn_factory_path_variant_model=fn_model_creation_log,
        max_nbr_paths=400, max_nbr_variants=400,
        fix_paths=True, fix_log_lang=True)

    print('Initial weights log domain')
    print(spn_wo_method_log.transition_weights)
    optimizer = tf.keras.optimizers.Adam(0.001)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    (updated_weights, error_series, training_time) = spn_wo_method_log.run_training_regime_simple(optimizer, 3000)
    print(updated_weights)
    #spn_container.update_transition_weights(updated_weights)
    #spn_container.export_to_file('test-log-logcalc.pnml')

    #spn_container.update_transition_weights(updated_weights)

