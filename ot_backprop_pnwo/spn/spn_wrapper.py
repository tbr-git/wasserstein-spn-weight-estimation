import numpy as np
import numpy.typing as npt
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

from ..stochastic_language.actindexing import ActivityIDConnector

class SPNWrapper:

    def __init__(self, act_id_connector: ActivityIDConnector, net, im, fm) -> None:
        self.net = net
        self.im = im
        self.fm = fm

        # Mapping transitions to
        # -> Ids (Indexes)
        # -> Activity Ids
        self.t_id_act_conn = TransitionIdActivityConnector(act_id_connector, net)

        self._initial_weights = self.get_weights()

    def update_transition_weights(self, updated_weights: npt.NDArray[np.int_]) -> None:
        for i, w in enumerate(updated_weights):
            self.t_id_act_conn.get_transition(i).properties['stochastic_distribution'].random_variable.weight = w
        
    def export_to_file(self, path: str):
        pnml_exporter.apply(self.net, self.im, path, final_marking=self.fm)

    def get_weights(self) -> npt.NDArray[np.float_]:
        return np.array([t.properties['stochastic_distribution'].random_variable.weight 
                         for t in self.t_id_act_conn.tId_2_transition])

    @property
    def initial_weights(self) -> npt.NDArray[np.float_]:
        return self._initial_weights

    @property
    def nbr_transitions(self) -> int:
        return len(self._initial_weights)


class TransitionIdActivityConnector:

    def __init__(self, act_id_connector: ActivityIDConnector, net):
        self.act_id_connector = act_id_connector

        self.transition_2_id = {}
        tId_2_transition = list()
        tId_2_actId = list()

        t_id = 0
        for t in net.transitions:
            self.transition_2_id[t] = t_id
            tId_2_transition.append(t)
            if t.label is not None:
                tId_2_actId.append(act_id_connector.get_activity_id(t.label))
            else:
                tId_2_actId.append(-1)

            t_id += 1

        self.tId_2_transition = np.array(tId_2_transition)
        self.tId_2_actId = np.array(tId_2_actId)

    def get_transition_id(self, t):
        return self.transition_2_id[t]

    def get_transition(self, t_id: int):
        return self.tId_2_transition[t_id]

    def get_activity_id(self, t_id: int):
        return self.tId_2_actId[t_id]

    def get_activity(self, t_id: int):
        return self.act_id_connector.get_activity(self.tId_2_actId[t_id])

    #def get_transition_weight(self, t_id: int):
    #    self.tId_2_transition[t_id].properties['stochastic_distribution'].random_variable.weight

    #def get_transition_weights(self):
    #    return np.array([t.properties['stochastic_distribution'].random_variable.weight for t in self.tId_2_transition])