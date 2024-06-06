from collections import namedtuple
from enum import Enum
from functools import partial
import logging
import pm4py
from typing import Callable, Optional
import random
import time
import heapq
import numpy as np
import itertools
import sys
from typing import Dict, List, Optional, Tuple, Union

from pm4py.objects.petri_net.semantics import weak_execute as execute_transition
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.semantics import ClassicSemantics

from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper

from .stochastic_path import StochasticPath
from .spn_util import get_transition_weight

logger = logging.getLogger(__name__)

SAMPLE_HEURISTIC_BOUNDED_RESIDUAL_MIN_POOL_SIZE = 2000
SAMPLE_HEURISTIC_BOUNDED_RESIDUAL_FACTOR_TARGET = 5
SAMPLING_FALLBACK_PROBING_COLLISION_STREAK = 150
SPN_TRANSITION_PROBABILITY = Enum('SPN_TRANSITION_PROBABILITY', ['UNIFORM', 'TRANSITION_WEIGHT'])

FILTER_PATHS_LONG_TRANSITION_FACTOR = 2
FILTER_PAHTS_LOW_PROBABILITY = 10 ** (-6)


SinglePathSample = namedtuple('SinglePathSample', 'probed_sampling nbr_paths_full nbr_paths_robustified paths_robustified')


def get_robust_path_sample(spn_container: SPNWrapper, nbr_samples: int, 
                   sample_size: int, trans_lh: SPN_TRANSITION_PROBABILITY=SPN_TRANSITION_PROBABILITY.UNIFORM) -> tuple[tuple[StochasticPath]]:
                
    logger.debug("Creating dataset of path samples")


    spn_paths_samples = sample_paths_heuristic_bounded_residual(spn_container=spn_container, 
                                                                transition_likelihood=trans_lh, 
                                                                target_sample_size=sample_size, 
                                                                nbr_samples=nbr_samples)

    # Filter path sample; remove super low probability but rediculous long paths
    spn_robustified_paths_samples = tuple(map(partial(robustify_path_sample_for_optimization, spn_container), spn_paths_samples))
    nbr_removed_paths = sum(len(p_sample_1) - len(p_sample_2) for p_sample_1, p_sample_2 in zip(spn_paths_samples, spn_robustified_paths_samples))
    if nbr_removed_paths > 0:
        logger.debug(f"Removed {nbr_removed_paths} while robustifying path samples.")
    return spn_robustified_paths_samples

    
def get_single_robust_path_sample(spn_container: SPNWrapper, 
                   sample_size: int, trans_lh: SPN_TRANSITION_PROBABILITY=SPN_TRANSITION_PROBABILITY.UNIFORM,
                   max_time_till_fallback_ms=30_000) -> SinglePathSample:
    ### Stochastic Path Language
    # Try to sample nbr paths + 1
    # Sampling prefers "short" paths
    if trans_lh is SPN_TRANSITION_PROBABILITY.UNIFORM:
        #path_pool = sample_most_likely_net_paths(spn_container=spn_container, assume_uniform_weights=True, max_num_paths=target_pool_size, max_total_time_ms=30_000)
        (could_unfold, path_sample) = sample_most_likely_spn_paths_with_fallback(spn_container=spn_container, 
                                                                               assume_uniform_weights=True, 
                                                                               max_nbr_paths=sample_size, 
                                                                               max_time_till_fallback_ms=max_time_till_fallback_ms)
    elif trans_lh is SPN_TRANSITION_PROBABILITY.TRANSITION_WEIGHT:
        #path_pool = sample_most_likely_net_paths(spn_container=spn_container, assume_uniform_weights=False, max_num_paths=target_pool_size, max_total_time_ms=30_000)
        (could_unfold, path_sample) = sample_most_likely_spn_paths_with_fallback(spn_container=spn_container, 
                                                                               assume_uniform_weights=False, 
                                                                               max_nbr_paths=sample_size, 
                                                                               max_time_till_fallback_ms=max_time_till_fallback_ms)
    else:
        raise Exception(f"Unknown transition likelihood {trans_lh}")

    path_sample_robustified = robustify_path_sample_for_optimization(spn_container, path_sample)
    result = SinglePathSample(not could_unfold, len(path_sample), len(path_sample_robustified), path_sample_robustified)
    return result


def sample_paths_heuristic_bounded_residual(spn_container: SPNWrapper,
                       transition_likelihood: SPN_TRANSITION_PROBABILITY,
                       target_sample_size: int, nbr_samples=1) -> tuple[tuple[StochasticPath]]:
    """Creates a path sample where we "heuristically" sample without replacement a target number of paths from the set of paths.
    
    Since the set of paths can be infinite, we employ a simple heuristic:
    1. We create a "large" pool of most likely paths 
        max(SAMPLE_HEURISTIC_BOUNDED_RESIDUAL_MIN_POOL_SIZE, SAMPLE_HEURISTIC_BOUNDED_RESIDUAL_FACTOR_TARGET * target_sample_size)
    2. Sample "nbr_samples" samples without replacement from the pool

    The probability of each paths not contained in the pool is bounded by 1/ pool size. 
    Therefore, the approximation should be quite ok for large pool sizes. 

    Args:
        spn_container (SPNWrapper): SPN Container
        transition_likelihood (SPN_TRANSITION_PROBABILITY): Probability of a transition given a set of enabled transition
        target_sample_size (int): Number of paths contained in a sample (less than the given number, if number exceeds number of paths)
        nbr_samples (int): How many sample to create (re-use pool). Defaults to 1.

    Returns:
        tuple[tuple[StochasticPath]]: Given number of path samples
    """
    ########################################
    # Idea for a precise sampling method:
    # Create pool P "\setunion"  [1 / target_sample_size]^{target_sample size} (residual pool R)
    # Do:
    #   Sample from extended pool under consideration of the paths' likelihoods. 
    #   Save non-residual samples
    #   Further unfold SPN and create P' (like above) using NEW paths only
    # While (sample from residual pool R is contained)
    # If we assume that we re-weight the probabilities in P' by the residual (not required in implementation because each path would have same normalization factor), 
    # this should create a real sample of the net where we terminate with probability 1.
    ########################################
    # Create pool
    target_pool_size = max(SAMPLE_HEURISTIC_BOUNDED_RESIDUAL_FACTOR_TARGET * target_sample_size, 
                           SAMPLE_HEURISTIC_BOUNDED_RESIDUAL_MIN_POOL_SIZE)
    if transition_likelihood is SPN_TRANSITION_PROBABILITY.UNIFORM:
        #path_pool = sample_most_likely_net_paths(spn_container=spn_container, assume_uniform_weights=True, max_num_paths=target_pool_size, max_total_time_ms=30_000)
        (_, path_pool) = sample_most_likely_spn_paths_with_fallback(spn_container=spn_container, 
                                                                               assume_uniform_weights=True, 
                                                                               max_nbr_paths=target_pool_size, 
                                                                               max_time_till_fallback_ms=30_000)
    elif transition_likelihood is SPN_TRANSITION_PROBABILITY.TRANSITION_WEIGHT:
        #path_pool = sample_most_likely_net_paths(spn_container=spn_container, assume_uniform_weights=False, max_num_paths=target_pool_size, max_total_time_ms=30_000)
        (_, path_pool) = sample_most_likely_spn_paths_with_fallback(spn_container=spn_container, 
                                                                               assume_uniform_weights=False, 
                                                                               max_nbr_paths=target_pool_size, 
                                                                               max_time_till_fallback_ms=30_000)
    else:
        raise Exception(f"Unknown transition likelihood {transition_likelihood}")

    path_pool_as_array = np.asarray(path_pool)
    path_pool_weights = np.array([p.probability for p in path_pool])
    # Normalize to obtain sampling probabilities
    path_pool_weights = path_pool_weights / np.sum(path_pool_weights)
    target_sample_size = min(target_sample_size, len(path_pool_weights))
    res = tuple(np.random.choice(path_pool_as_array, p=path_pool_weights, size=target_sample_size, replace=False) for _ in range(nbr_samples))

    #path_pool_weights = np.array([p.probability for p in path_pool])
    #target_sample_size = min(target_sample_size, len(path_pool_weights))
    #res = tuple(random.choices(path_pool, weights=path_pool_weights, k=target_sample_size, ) for _ in range(nbr_samples))
    return res
    

def sample_most_likely_spn_paths_with_fallback(
        spn_container: SPNWrapper, 
        assume_uniform_weights: bool = False,
        max_nbr_paths: Optional[int] = None, 
        max_time_till_fallback_ms: Optional[int] = None
) -> tuple[bool, tuple[StochasticPath]]:
    """
    Attempts to sample the most likely paths either using the current weights or assuming uniform weights.
    If the SPN cannot be unfolded easily, the fallback solution is probing.
    """
    paths = sample_most_likely_net_paths(spn_container, 
                                         assume_uniform_weights=assume_uniform_weights, 
                                         max_num_paths=max_nbr_paths, 
                                         max_total_time_ms=max_time_till_fallback_ms)
    # If this fails, "BFS" exploration might branch so hard, that we cannot get any path
    # Probing "DFS" Sampling
    if paths is None or len(paths) == 0:
        if assume_uniform_weights:
          sampling_method = random_transition_choice
        else:
          sampling_method = random_weighted_transition_choice
        logger.debug("Could not sample most likely paths (BFS), fallback to probing")
        paths_fallback = sample_paths_collision_heuristic(spn_container, 
                                                          sampling_method, max_nbr_paths, 
                                                          SAMPLING_FALLBACK_PROBING_COLLISION_STREAK)
        # If fallback retrieves even fewer
        if paths is not None and len(paths) >= len(paths_fallback):
            return (True, paths)
        return (False, paths_fallback)

    return (True, paths)


def sample_paths_collision_heuristic(net_container: SPNWrapper,
                       fn_transition: Callable[[tuple[pm4py.PetriNet.Transition]], pm4py.PetriNet.Transition],
                       target_nbr: int, stop_collision_streak: int) -> tuple[StochasticPath]:
    l_paths = list()
    nbr_paths = 0
    collision_streak = 0

    while collision_streak < stop_collision_streak and nbr_paths < target_nbr:
        p = sample_single_path(net_container, fn_transition)

        if p not in l_paths:
            l_paths.append(p)
            collision_streak = 0
            nbr_paths += 1
        else:
            collision_streak += 1

    return tuple(l_paths)


################################################################################
# Base Sampling
################################################################################
def sample_single_path(net_container: SPNWrapper,
                       fn_transition: Callable[[tuple[pm4py.PetriNet.Transition]], pm4py.PetriNet.Transition]
                       ) -> StochasticPath:

    net = net_container.net
    t_id_act_conn = net_container.t_id_act_conn
    m = net_container.im
    # Path of transitions
    path = list()
    # Enabled transitions along path
    path_enabled = list()
    enabled_transitions = tuple(pm4py.get_enabled_transitions(net, m))
    # Initial probability
    p = 1

    ##### Sampling Loop #####
    while len(enabled_transitions) > 0:
        t, p_step = fn_transition(enabled_transitions)
        p *= p_step

        # Save step (translated into ids)
        path.append(t_id_act_conn.get_transition_id(t))
        path_enabled.append(
            np.array(tuple(map(
                t_id_act_conn.get_transition_id, 
                enabled_transitions
            )), dtype=np.int32)
        )

        # Execute step
        m = execute_transition(t, m)
        enabled_transitions = tuple(pm4py.get_enabled_transitions(net, m))

    stoch_path = StochasticPath(np.array(path, dtype=np.int32), tuple(path_enabled), p, t_id_act_conn)
    return stoch_path


########################################
# Selecting Transitions
########################################
def random_transition_choice(enabled_transitions: tuple[pm4py.PetriNet.Transition]) -> tuple[pm4py.PetriNet.Transition, float]:
    """Randomly samples a transition from the given transitions

    Uniformly at random from the set of transitions (weights are not considered).

    Args:
        enabled_transitions (tuple[pm4py.PetriNet.Transition]): Collection of transitions (must not be empty)

    Returns:
        pm4py.PetriNet.Transition: transition
        float: Probability of this step
    """
    t = random.sample(enabled_transitions, 1)[0]
    p = get_transition_weight(t) / sum(map(get_transition_weight, enabled_transitions))
    return t, p


def random_weighted_transition_choice(enabled_transitions: tuple[pm4py.PetriNet.Transition]) -> tuple[pm4py.PetriNet.Transition, float]:
    """Random weighted sampling of a transition 

    Args:
        enabled_transitions (tuple[pm4py.PetriNet.Transition]): Collection of transitions (must not be empty)

    Returns:
        pm4py.PetriNet.Transition: transition
        float: Probability of this step
    """
    t_weights = tuple(map(get_transition_weight, enabled_transitions))
    t = random.choices(enabled_transitions, weights=t_weights, k=1)[0]
    p = get_transition_weight(t) / sum(t_weights)
    return t, p


########################################
# Unfolding GSPNs
########################################
def sample_most_likely_net_paths(
        spn_container: SPNWrapper, 
        assume_uniform_weights: bool = False,
        prob_mass: Optional[float] = None,
        max_num_paths: Optional[int] = None, 
        max_total_time_ms: Optional[int] = None
) -> tuple[StochasticPath]:
    """
    Samples the most likely paths from the Petri net until one of the stop criteria is reached.
    If a parameter is None, then it's not considered.

    !!! This method is copied from Eduardo Goulart Rocha's code!!!
    """
    assert prob_mass is not None or max_num_paths is not None or max_total_time_ms is not None, \
        "At least one of the stopping criteria must be specified."

    if prob_mass is None:
        # Use a mass strictly greater than 1
        # Avoids termination due to numerical issues for low probability paths
        prob_mass = 1.1
    if max_num_paths is None:
        max_num_paths = sys.maxsize
    if max_total_time_ms is None:
        max_total_time_ms = sys.maxsize
    max_total_time_ns = max_total_time_ms * 1_000_000

    net, im = spn_container.net, spn_container.im

    paths = []
    sampled_mass = 0.0
    semantics = ClassicSemantics()
    marking_transition_cache: Dict[Marking, List[Tuple[PetriNet.Transition, Marking, float]]] = {}

    # heapq is one of the few situations where C++ is easier to use than Python
    # We use tuples because the overhead of class
    to_explore = []
    # To break ties in the queue
    id_counter = itertools.count(0)

    # path_id -> (incoming_transition, prev_path)
    path_transition_function = {}

    def _push_heap(_prob, _prev_path_id, _input_transition, _marking):
        path_id = next(id_counter)
        # Keep the heap compact by storing other data outside
        path_transition_function[path_id] = _prev_path_id, _input_transition
        # Must invert the probability to get a max-heap
        heapq.heappush(to_explore, (-_prob, path_id, _marking))

    start_time = time.time_ns()

    def _should_terminate():
        return sampled_mass >= prob_mass or len(paths) >= max_num_paths \
            or (time.time_ns() - start_time) > max_total_time_ns

    _push_heap(1.0, None, None, im)
    it_counter = itertools.count(0)
    while to_explore:
        it_count = next(it_counter)
        if it_count % 100 == 0 and _should_terminate():
            break
        if it_count % 10000 == 0:
            logger.debug("Mass: %f - Paths: %d - Pending: %d", sampled_mass, len(paths), len(to_explore))

        cur_prob, cur_path_id, cur_marking = heapq.heappop(to_explore)
        cur_prob = -cur_prob

        if cur_marking not in marking_transition_cache:
            enabled_transitions = semantics.enabled_transitions(net, cur_marking)
            if assume_uniform_weights:
                weight_sum = len(enabled_transitions)
                marking_transition_cache[cur_marking] = [
                    (transition, semantics.weak_execute(transition, net, cur_marking),
                    1 / weight_sum)
                    for transition in enabled_transitions
                ]
            else:
                weight_sum = sum(get_transition_weight(transition) for transition in enabled_transitions)
                marking_transition_cache[cur_marking] = [
                    (transition, semantics.weak_execute(transition, net, cur_marking),
                    get_transition_weight(transition) / weight_sum)
                    for transition in enabled_transitions
                ]

        transition_function = marking_transition_cache[cur_marking]
        if len(transition_function) == 0:
            # Reconstruct path
            path = []
            prev_path_id, prev_transition = path_transition_function[cur_path_id]
            while prev_transition is not None:
                path.append(prev_transition)
                prev_path_id, prev_transition = path_transition_function[prev_path_id]
            path.reverse()
            paths.append((path, cur_prob))
            sampled_mass += cur_prob
            continue

        for transition, tgt_marking, transition_prob in transition_function:
            tgt_prob = cur_prob * transition_prob
            # In case of loops, this assertion becomes problematic due to numerical problems
            # assert tgt_prob != 0.0
            _push_heap(tgt_prob, cur_path_id, transition, tgt_marking)

    distinct_paths = {''.join(transition.name for transition in path) for path, _ in paths}
    assert len(distinct_paths) == len(paths), \
        "Cannot sample the same path twice. This indicates an error in the logic"

    path_probs = [path_prob for _, path_prob in paths]
    assert all(prev_prob >= next_prob for prev_prob, next_prob in zip(path_probs, path_probs[1:])), \
        "Paths should be generated from most to least likely"

    ##############################
    # Create Stochastic Paths with enabled transition information and
    # index-based encoding of transition 
    ##############################
    l_paths: List[StochasticPath] = []
    for (path, prob) in paths:
        l_transition_ids: List[int] = list()
        l_enabled_transition_ids: List[Tuple[int]] = list()
        cur_marking = im
        for t in path:
            l_transition_ids.append(spn_container.t_id_act_conn.get_transition_id(t))
            enabled_transitions = semantics.enabled_transitions(net, cur_marking)
            l_enabled_transition_ids.append(np.array(tuple(
                spn_container.t_id_act_conn.get_transition_id(t2) for t2 in enabled_transitions
            ), dtype=np.int32))
            cur_marking = semantics.weak_execute(t, net, cur_marking)
        l_paths.append(StochasticPath(np.array(l_transition_ids, dtype=np.int32), tuple(l_enabled_transition_ids), prob, spn_container.t_id_act_conn))
    my_stochastic_paths = tuple(l_paths)
    logger.debug(f"Eduardo paths found: {len(paths)}")

    return my_stochastic_paths


################################################################################
# Help Util
################################################################################
def robustify_path_sample_for_optimization(spn_container: SPNWrapper, path_sample: tuple[StochasticPath]) -> tuple[StochasticPath]:
    """Shared funtionality to robustify a path sample - that is, remove POTENTIALLY (not verified) problematic outlier behavior. 

    This removes:
    - Extremely long, low probability paths.

    Args:
        spn_container (SPNWrapper): SPN (information can be used to drop paths from sample)
        path_sample (tuple[StochasticPath]): Sample to process

    Returns:
        tuple[StochasticPath]: Path sample where potentially problematic paths have been dropped
    """
    spn_container.nbr_transitions

    def keep_path_condition(path: StochasticPath) -> bool:
        if len(path) > FILTER_PATHS_LONG_TRANSITION_FACTOR  * spn_container.nbr_transitions and path.probability < FILTER_PAHTS_LOW_PROBABILITY:
            return False
        return True

    return tuple(filter(keep_path_condition, path_sample))

##############################
# Path Statistics
##############################
def log_sample_statistics(path_samples: tuple[tuple[StochasticPath]]):
    if logger.isEnabledFor(logging.DEBUG):
        sample_statistics = collect_sample_statistics(path_samples)
        logger.debug(str(sample_statistics))


def collect_sample_statistics(path_samples: tuple[tuple[StochasticPath]]):
    sample_statistics = tuple(map(compute_path_sample_statistics, path_samples))
    return sample_statistics


def compute_path_sample_statistics(path_sample: tuple[StochasticPath]):
    probabilities = np.fromiter(map(StochasticPath.probability.fget, path_sample), np.float32)
    probability_sum = np.sum(probabilities)
    probability_min = np.min(probabilities)
    probability_max = np.max(probabilities)
    nbr_paths = len(probabilities)

    return {
        'probabilitySum': probability_sum,
        'probabilityMin': probability_min,
        'probabilityMax': probability_max,
        'nbrPaths': nbr_paths,
    }
