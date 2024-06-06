import pandas as pd
import numpy as np
import numpy.typing as npt

class StochasticPath:

    def __init__(self, transition_ids: npt.NDArray[np.int_], curr_enabled_transitions: tuple[npt.NDArray[np.int_]], probability: float, tIdActConn):
        self.transition_ids = transition_ids
        self.curr_enabled_transitions = curr_enabled_transitions
        self.tIdActConn = tIdActConn
        self._probability = probability
        
    def getVisibleActivityIds(self):
        return list(filter(lambda a_id : a_id is not None and a_id >= 0, map(self.tIdActConn.get_activity_id, self.transition_ids)))

    @property
    def probability(self):
        return self._probability

    def __len__(self):
        return len(self.transition_ids)

    def __hash__(self):
        pd.util.hash_array(self.transition_ids)

    def __eq__(self, other):
        if isinstance(other, StochasticPath):
            return np.array_equal(self.transition_ids, other.transition_ids)
        return NotImplemented

    def __ne__(self, other):
        x = self.__eq__(other)
        if x is NotImplemented:
            return NotImplemented
        return not x

    def __str__(self):
        l_str_path = []
        for t_id, t_enabled in zip(self.transition_ids, self.curr_enabled_transitions):
            str_enabled = ", ".join(f"({t_id2}, {self.tIdActConn.get_activity(t_id2)})" for t_id2 in t_enabled)
            l_str_path.append(f"[({t_id}, {self.tIdActConn.get_activity(t_id)})-{{{str_enabled}}}]")
        return "<" + ",".join(l_str_path) + ">"
