import pm4py
import numpy as np
import numpy.typing as npt
import pandas as pd
import random
from .stochastic_language.actindexing import ActivityIDConnector
from .stochastic_language.stochastic_lang import StochasticLang
from pm4py.objects.petri_net.semantics import enabled_transitions
from pm4py.objects.petri_net.semantics import weak_execute as execute_transition

class PNConnectorOT:

    STOP_SAMPLING_IF_NOTHING_NEW = 100

    def __init__(self, net, initial_marking, act_id_connector: ActivityIDConnector, target_nbr_path: int):
        self.act_id_connector = act_id_connector
        self.net = net
        self.initial_marking = initial_marking
        self.target_nbr_path = target_nbr_path

        self.tIdActConn = TransitionIdActivityConnector(act_id_connector, net)

        self.paths = self.sample_paths()
        (self.stoch_lang, self.lang_2_path) = StochasticLang.from_stochastic_paths(self.paths, act_id_connector)


    def sample_paths(self):

        l_paths = list()
        l_enabled_along_path = list()
        l_probabilities = list()
        nbr_paths = 0 
        collision_streak = 0

        while collision_streak < PNConnectorOT.STOP_SAMPLING_IF_NOTHING_NEW and nbr_paths < self.target_nbr_path:
            path, enabled_along_path, p = self.sample_single_path()

            if path not in l_paths:
                l_paths.append(path)
                l_enabled_along_path.append(enabled_along_path)
                l_probabilities.append(p)
                collision_streak = 0
                nbr_paths += 1
            else:
                collision_streak += 1

        res = tuple(StochasticPath(np.array(p), tuple(np.array(t_enabed) for t_enabed in enabled_along_path), prob, self.tIdActConn) 
            for p, enabled_along_path, prob in zip(l_paths, l_enabled_along_path, l_probabilities))

        return res

    def sample_single_path(self):
        path = list()
        curr_enabled_transitions = list()
        m = self.initial_marking

        t_enabled = tuple(enabled_transitions(self.net, m))
        p = 1

        while len(t_enabled) > 0:
            t = random.sample(t_enabled, 1)[0]

            # TODO
            # IMPORTANT: Assume all transition weights are 1
            p *= 1 / len(t_enabled)

            path.append(t)
            curr_enabled_transitions.append(t_enabled)

            m = execute_transition(t, m)
            t_enabled = tuple(enabled_transitions(self.net, m))

        ####################
        # Translate into ids
        ####################
        path = list(map(self.tIdActConn.get_transition_id, path))
        curr_enabled_transitions = list([list(map(self.tIdActConn.get_transition_id, t_en))  for  t_en in curr_enabled_transitions])

        return (path, curr_enabled_transitions, p)


class StochasticPath:

    def __init__(self, transition_ids: npt.NDArray[np.int_], curr_enabled_transitions: tuple[npt.NDArray[np.int_]], probability: float, tIdActConn):
        self.transition_ids = transition_ids
        self.curr_enabled_transitions = curr_enabled_transitions
        self.tIdActConn = tIdActConn
        self.probability = probability
        
    def getVisibleActivityIds(self):
        return list(filter(lambda a_id : a_id is not None and a_id >= 0, map(self.tIdActConn.get_activity_id, self.transition_ids)))

    def __hash__(self):
        pd.util.hash_array(self.transition_ids)

    def __eq__(self, other):
        if isinstance(other, StochasticPath):
            np.array_equal(self.transition_ids, other.transition_ids)

    def __str__(self):
        l_str_path = []
        for t_id, t_enabled in zip(self.transition_ids, self.curr_enabled_transitions):
            str_enabled = ", ".join(f"({t_id2}, {self.tIdActConn.get_activity(t_id2)})" for t_id2 in t_enabled)
            l_str_path.append(f"[({t_id}, {self.tIdActConn.get_activity(t_id)})-{{{str_enabled}}}]")
        return "<" + ",".join(l_str_path) + ">"

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

    def get_transition_weight(self, t_id: int):
        self.tId_2_transition[t_id].properties['stochastic_distribution'].random_variable.weight

    def get_transition_weights(self):
        return np.array([t.properties['stochastic_distribution'].random_variable.weight for t in self.tId_2_transition])

