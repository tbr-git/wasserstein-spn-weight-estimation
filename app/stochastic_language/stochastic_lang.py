import logging
import pm4py
import numpy as np
import numpy.typing as npt
import pandas as pd
import numba

from ..spn.stochastic_path import StochasticPath
from .actindexing import ActivityIDConnector

logger = logging.getLogger(__name__)


class StochasticLang:

    def __init__(self, act_id_connector: ActivityIDConnector, variants: list[npt.NDArray[np.int_]], 
                 probabilities: npt.NDArray[np.float_]):
        self.act_id_connector = act_id_connector
        self.variants = variants
        self.probabilities = probabilities

    def random_subsample(self, nbr_variants: int):
        nbr_variants = min(nbr_variants, len(self.variants))
        var_indices = np.random.choice(len(self.variants), size=nbr_variants, replace=False,
                                p=self.probabilities)
        
        p_sample = np.array([self.probabilities[i] for i in var_indices], np.float64)
        variants_sample = [self.variants[i] for i in var_indices]

        return (variants_sample, p_sample)

    def most_likely_variants(self, nbr_variants):
        if len(self.variants) <= nbr_variants:
            return (self.variants, self.probabilities)
        else:
            index_sorted = np.argsort(self.probabilities)
            index_sorted = index_sorted[-nbr_variants::]

            p_sample = self.probabilities[index_sorted]
            variants_sample = [self.variants[i] for i in index_sorted]
            return (variants_sample, p_sample)

    def from_stochastic_paths(paths: tuple[StochasticPath], act_id_connector: ActivityIDConnector):
        l_variants = list()
        l_weights = list()
        #stoch_path_2_variant = np.zeros(len(paths))
        variant_2_stoch_paths = dict()

        for i, stoch_path in enumerate(paths):
            v = stoch_path.getVisibleActivityIds()
            if v in l_variants:
                v_index = l_variants.index(v)
                l_weights[v_index] = l_weights[v_index] + stoch_path.probability
                variant_2_stoch_paths[v_index].append(i)
            else:
                l_variants.append(v)
                l_weights.append(stoch_path.probability)
                variant_2_stoch_paths[len(l_variants) - 1] = [i]

        probabilities = np.array(l_weights, dtype=np.float64)
        variants = list(np.array(v, dtype=np.int32) for v in l_variants)

        return (StochasticLang(act_id_connector, variants, probabilities), variant_2_stoch_paths)

    def from_event_log(df: pd.DataFrame, act_id_connector: ActivityIDConnector):
        pm4py_variants = pm4py.get_variants(df)

        l_variants = list()
        weights = np.zeros(len(pm4py_variants), dtype=np.float64)
        nbr_traces = 0

        for i, (v, nbr_traces_variant) in enumerate(pm4py_variants.items()):
            l_variants.append(np.array(list(map(act_id_connector.get_activity_id, v)), dtype=np.int32))
            weights[i] = nbr_traces_variant
            nbr_traces += nbr_traces_variant

        weights /= nbr_traces

        return StochasticLang(act_id_connector, l_variants, weights)

    def __str__(self):
        l_str_variants = []
        for v, p in zip(self.variants, self.probabilities):
            l_str_variants.append(f"{p}: <{', '.join(map(self.act_id_connector.get_activity, v))}>")
        return "{" + "\n".join(l_str_variants) + "}"

    def __len__(self):
        return len(self.variants)


@numba.njit
def _lvs_cost_matrix(
        lang1: numba.typed.List[np.array], 
        lang2: numba.typed.List[np.array]):
    c = np.empty(shape=(len(lang1), len(lang2)))
    for i, trace1 in enumerate(lang1):
        for j, trace2 in enumerate(lang2):
            c[i, j] = post_normalized_lvs(trace1, trace2)
            pass
    return c


def lvs_cost_matrix(lang1: StochasticLang, lang2: StochasticLang):
    try:
        numba_lang1 = numba.typed.List(lang1.variants)
        numba_lang2 = numba.typed.List(lang2.variants)

        c = _lvs_cost_matrix(numba_lang1, numba_lang2)
    except Exception as err:
        logger.error("Problem in cost computation matrix with model {%s} --- lang {%s}", str(lang1), str(lang2)) 
        raise
    return c


@numba.njit
def post_normalized_lvs(left: npt.NDArray[np.int_], right: npt.NDArray[np.int_]):
    n = len(left)
    m = len(right)

    lvs = _compiled_lvs(left, right)
    return lvs / max(n, m)


@numba.njit
def _compiled_lvs(left: npt.NDArray[np.int_], right: npt.NDArray[np.int_]):
    n = len(left)
    m = len(right)

    if n == 0: 
        return m
    elif m == 0:
        return n

    if n > m:
        # swap the input strings to consume less memory
        tmp = left
        left = right
        right = tmp
        n = m
        m = len(right)

    p = np.arange(n + 1)

    for j in range(1, m + 1):
        upper_left = p[0]
        rightJ = right[j - 1]
        p[0] = j

        for i in range(1, n + 1):
            upper = p[i]
            cost = 0 if left[i - 1] == rightJ else 1
            # minimum of cell to the left+1, to the top+1, diagonally left and up +cost
            p[i] = min(p[i - 1] + 1, p[i] + 1, upper_left + cost)
            upper_left = upper

    return p[n]
