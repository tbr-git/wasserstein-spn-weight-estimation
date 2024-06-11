from dataclasses import dataclass

from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType


@dataclass
class ConvergenceConfig:
    nbr_iterations_min: int
    nbr_iterations_max: int
    eps_convergence:float


@dataclass
class TwoPhaseRunUniqueIdentifyingConfig:
    """Unique configuration that identifies a run based on the parameter.
    Given two runs using this configuration, only random initialization or the iteration "id" 
    distinguishes the runs
    """
    emsc_loss_type: EMSCLossType
    name_log: str
    name_spn: str
    max_nbr_paths: int
    max_nbr_variants: int
    hot_start: bool
    iteration: int
    phase_two_enabled: bool
    convergence_config: ConvergenceConfig 

    