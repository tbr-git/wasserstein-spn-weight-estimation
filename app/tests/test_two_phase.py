import unittest
import pandas as pd
import logging
import tensorflow as tf

import pm4py

from ot_backprop_pnwo.optimization.ot_wo_two_phase import OT_WO_Two_Phase
from ot_backprop_pnwo.spn import spn_util
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.stochastic_language.actindexing import ActivityIDConnector
from ot_backprop_pnwo.stochastic_language.stochastic_lang import StochasticLang


class TestPhaseTwo(unittest.TestCase):

    LOG_PATH = '/home/tobias/ot-wo-evaluation/data/logs/road_fines/RTFM_as_CSV.csv'

    MODEL_PATH_SMALL = '/home/tobias/ot-wo-evaluation/data/models/road_fines/cold-start/rtfm_HM_ABE.pnml'

    MODEL_PATH_SMALL_2 = '/home/tobias/ot-wo-evaluation/data/models/road_fines/cold-start/rtfm_DFM_ABE.pnml'

    MODEL_PATH_INFINITE = '/home/tobias/ot-wo-evaluation/data/models/road_fines/cold-start/rtfm_IMf02_ABE.pnml'

    def test_optimization_two_phase_fix_2(self):
        ### Event Log
        logging.info("Importing log")
        df_ev = pd.read_csv(TestPhaseTwo.LOG_PATH, sep=',')
        df_ev = pm4py.format_dataframe(df_ev, case_id='case', activity_key='event', timestamp_key='completeTime')

        ##### Load Net and Event Log Side #####
        # SPN
        net, im, fm = spn_util.load_as_spn(TestPhaseTwo.MODEL_PATH_SMALL_2)

        act_id_conn = ActivityIDConnector(df_ev, net)
        spn_container = SPNWrapper(act_id_conn, net, im, fm)
        stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)

        spn_wo_alg = OT_WO_Two_Phase(spn_container=spn_container, stoch_lang_log=stoch_lang_log, act_id_conn=act_id_conn,
                                     hot_start=False, run_phase_two=True)

        optimizer = tf.keras.optimizers.Adam(0.001)
        spn_wo_alg.optimize_weights(optimizer)

        # Model and log are small, should de-activate phase 2
        self.assertEqual(231, len(spn_wo_alg.data_phase_1.pseudo_log_lang))
        self.assertEqual(3, len(spn_wo_alg.data_phase_1.spn_paths))
        self.assertEqual(3, len(spn_wo_alg.data_phase_1.pseudo_spn_lang))
        self.assertFalse(spn_wo_alg._run_phase_two)


    def _test_optimization_two_phase_fix(self):
        ### Event Log
        logging.info("Importing log")
        df_ev = pd.read_csv(TestPhaseTwo.LOG_PATH, sep=',')
        df_ev = pm4py.format_dataframe(df_ev, case_id='case', activity_key='event', timestamp_key='completeTime')

        ##### Load Net and Event Log Side #####
        # SPN
        net, im, fm = spn_util.load_as_spn(TestPhaseTwo.MODEL_PATH_SMALL)

        act_id_conn = ActivityIDConnector(df_ev, net)
        spn_container = SPNWrapper(act_id_conn, net, im, fm)
        stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)

        spn_wo_alg = OT_WO_Two_Phase(spn_container=spn_container, stoch_lang_log=stoch_lang_log, act_id_conn=act_id_conn,
                                     hot_start=False, run_phase_two=True)

        optimizer = tf.keras.optimizers.Adam(0.001)
        spn_wo_alg.optimize_weights(optimizer)

        # Model and log are small, should de-activate phase 2
        self.assertFalse(spn_wo_alg._run_phase_two)

    def test_optimization_two_phase_sample_log(self):
        ### Event Log
        logging.info("Importing log")
        df_ev = pd.read_csv(TestPhaseTwo.LOG_PATH, sep=',')
        df_ev = pm4py.format_dataframe(df_ev, case_id='case', activity_key='event', timestamp_key='completeTime')

        ##### Load Net and Event Log Side #####
        # SPN
        net, im, fm = spn_util.load_as_spn(TestPhaseTwo.MODEL_PATH_SMALL)

        act_id_conn = ActivityIDConnector(df_ev, net)
        spn_container = SPNWrapper(act_id_conn, net, im, fm)
        stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)

        spn_wo_alg = OT_WO_Two_Phase(spn_container=spn_container, stoch_lang_log=stoch_lang_log, act_id_conn=act_id_conn,
                                     hot_start=False, run_phase_two=True,
                                     max_nbr_variants=50)

        optimizer = tf.keras.optimizers.Adam(0.001)
        spn_wo_alg.optimize_weights(optimizer)

        # Model and log are small, should de-activate phase 2
        self.assertTrue(spn_wo_alg._run_phase_two)

    
    def test_optimization_two_phase_sample_model(self):
        ### Event Log
        logging.info("Importing log")
        df_ev = pd.read_csv(TestPhaseTwo.LOG_PATH, sep=',')
        df_ev = pm4py.format_dataframe(df_ev, case_id='case', activity_key='event', timestamp_key='completeTime')

        ##### Load Net and Event Log Side #####
        # SPN
        net, im, fm = spn_util.load_as_spn(TestPhaseTwo.MODEL_PATH_INFINITE)

        act_id_conn = ActivityIDConnector(df_ev, net)
        spn_container = SPNWrapper(act_id_conn, net, im, fm)
        stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)

        spn_wo_alg = OT_WO_Two_Phase(spn_container=spn_container, stoch_lang_log=stoch_lang_log, act_id_conn=act_id_conn,
                                     hot_start=False, run_phase_two=True,
                                     max_nbr_variants=500)

        optimizer = tf.keras.optimizers.Adam(0.001)
        spn_wo_alg.optimize_weights(optimizer)

        # Model and log are small, should de-activate phase 2
        self.assertTrue(spn_wo_alg._run_phase_two)


    def test_optimization_two_phase_sample_model_sample_log(self):
        ### Event Log
        logging.info("Importing log")
        df_ev = pd.read_csv(TestPhaseTwo.LOG_PATH, sep=',')
        df_ev = pm4py.format_dataframe(df_ev, case_id='case', activity_key='event', timestamp_key='completeTime')

        ##### Load Net and Event Log Side #####
        # SPN
        net, im, fm = spn_util.load_as_spn(TestPhaseTwo.MODEL_PATH_INFINITE)

        act_id_conn = ActivityIDConnector(df_ev, net)
        spn_container = SPNWrapper(act_id_conn, net, im, fm)
        stoch_lang_log = StochasticLang.from_event_log(df_ev, act_id_conn)

        spn_wo_alg = OT_WO_Two_Phase(spn_container=spn_container, stoch_lang_log=stoch_lang_log, act_id_conn=act_id_conn,
                                     hot_start=False, run_phase_two=True,
                                     max_nbr_variants=50)

        optimizer = tf.keras.optimizers.Adam(0.001)
        spn_wo_alg.optimize_weights(optimizer)

        # Model and log are small, should de-activate phase 2
        self.assertTrue(spn_wo_alg._run_phase_two)


if __name__ == '__main__':
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    unittest.main()