import logging

import pm4py
from pm4py.util import constants
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.random_variables.random_variable import RandomVariable
from pm4py.objects.random_variables.uniform.random_variable import Uniform

from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper

module_logger = logging.getLogger(__name__)
# Compare with Path2VariantProbabilityLayer.MIN_INITIAL_TRANSITION_WEIGHT = 0.00001
# the layer again enforces a minimum weight; however, when sampling paths zero weights can also
# already cause a problem
MIN_TRANSITION_WEIGHT = 0.00001

def get_transition_weight(t: pm4py.PetriNet.Transition) -> float:
    """Get the weight of the given transtions (SPN) 

    Args:
        t (pm4py.PetriNet.Transition): Transition in SPN

    Returns:
        float: Weight of transition in SPN
    """
    return t.properties['stochastic_distribution'].random_variable.weight


def spn_allows_for_at_least_paths(spn_container: SPNWrapper, nbr_of_paths: int):
    """Determine whether an SPS allows at least for a certain number of paths.

    Try to sample the most likely given number + 1 paths (assuming unit weights) 
        
    Args:
        spn_container (SPNWrapper): SPN container
        nbr_of_paths (int): Number of paths 

    Returns:
        bool: Whether the SPN allows for at least the given number of paths
    """
    from ot_backprop_pnwo.spn.spn_path_sampling import sample_most_likely_net_paths
    stoch_paths = sample_most_likely_net_paths(spn_container, assume_uniform_weights=True, max_num_paths=nbr_of_paths + 1)
    return len(stoch_paths) < nbr_of_paths


def load_as_spn(path_pn):
    """Load the Petri net and add transition weights in case it is a normal Petri net

    Args:
        path_pn (pathlib.Path): Path to Petri net file

    Returns:
        tuple: (net, im, fm)
    """
    net, im, fm, stochastic_map = pnml_importer.apply(str(path_pn), parameters={"return_stochastic_map": True})
    
    # Add weight to transition if necessary
    for t in net.transitions:
        if not 'stochastic_distribution' in t.properties:
            rv = RandomVariable()
            rv.random_variable = Uniform()
            rv.set_weight(1)
            t.properties[constants.STOCHASTIC_DISTRIBUTION] = rv

    # No initial marking found
    # Set initial marking assuming that the Petri net is a workflow net
    if len(im) == 0:
        module_logger.warning("Initial marking of SPN is empty. Creating one under workflow net assumption.")
        for p in net.places:
            # No incoming arcs
            if len(p.in_arcs) == 0:
                im[p] += 1

        # Check (unexpected) initial markings after artificial creation
        if len(im) == 0:
            module_logger.warning(f"Failed to create an initial marking. No place without predecessor.  ww to createCreated initial marking {str(im)}")
        elif len(im) > 1:
            module_logger.warning(f"Multiple places without predecessor. Initial marking contains {str(len(im))} tokens")

    return (net, im, fm)


def ensure_non_zero_transition_weights(net):
    """Ensures that all transition weights are non-zero by enforcing a minimum weight

    Args:
        net Net whose transitions are adapted in place

    """
    for t in net.transitions:
        if t.properties['stochastic_distribution'].random_variable.weight == 0: 
            t.properties['stochastic_distribution'].random_variable.weight = MIN_TRANSITION_WEIGHT 

