import pm4py
import pandas as pd
import numpy as np

class ActivityIDConnector:

    def __init__(self, df_ev: pd.DataFrame, pn):

        ####################
        # Initialize activities from log
        ####################
        activities = list(df_ev['concept:name'].unique())
        self.act_2_id = {}

        for i, a in enumerate(activities):
            self.act_2_id[a] = i

        ####################
        # Integrate Petri net
        ####################
        for t in pn.transitions:
            # Skip silent
            if t.label is not None:
                if t.label not in self.act_2_id:
                    nextId = len(activities)
                    activities.append(t.label)
                    self.act_2_id[t.label] = nextId

        self.actId_2_activity = np.array(activities)

    def get_activity_id(self, activity: str):
        if activity in self.act_2_id:
            return self.act_2_id[activity]
        else:
            return -1

    def get_activity(self, id: int):
        if 0 <= id and id < len(self.actId_2_activity):
            return self.actId_2_activity[id]
        else:
            return None


