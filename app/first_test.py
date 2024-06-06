import pm4py
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
import pandas as pd
import numpy as np

from .stochastic_language.stochastic_lang import StochasticLang
from .stochastic_language.actindexing import ActivityIDConnector
from .pn_ot import PNConnectorOT
from .ot_weight_optimization_problem import OptimalTransportWeightOptimizer


if __name__ == "__main__":
    net, im, fm, stochastic_map = pnml_importer.apply("./test-rtfm.pnml", parameters={"return_stochastic_map": True})

    # Initial Marking
    for p in net.places:
        if p.name == 'n22':
            im[p] += 1

    # Event Log
    print("Importing log")
    df_ev = pd.read_csv('./RTFM_as_CSV.csv', sep=',')
    df_ev = pm4py.format_dataframe(df_ev, case_id='case', activity_key='event', timestamp_key='completeTime')

    actIdConn = ActivityIDConnector(df_ev, net)

    # Log language
    print("Calculating log stochastic language")
    stoch_lang_log = StochasticLang.from_event_log(df_ev, actIdConn)

    # Connect Petri net
    print("Connecting Petri Net")
    p_connector_ot = PNConnectorOT(net, im, actIdConn, 100)

    print("Optimization")
    test_ot_wo = OptimalTransportWeightOptimizer(p_connector_ot, stoch_lang_log)

    OptimalTransportWeightOptimizer.OPTIMIZATION_TIME_BOUND = 90 * 60

    error_series = test_ot_wo.optimize_weights()

    # Some result logging
    l_transitions = list(net.transitions)

    # Show weights before update
    #for t in l_transitions:
    #    print(t.properties['stochastic_distribution'].random_variable.weight)

    # Set updated weights
    for i, w in enumerate(test_ot_wo.trans_weights.numpy()):
        p_connector_ot.tIdActConn.get_transition(i).properties['stochastic_distribution'].random_variable.weight = w

    # Show weights after
    #for t in l_transitions:
    #    print(t.properties['stochastic_distribution'].random_variable.weight)

    # Export results
    #pnml_exporter.apply(net, im, "./test-update3.pnml", final_marking=fm)

    error_series = np.array(error_series)
    #np.savetxt("./error-series.csv", error_series, delimiter=",")

