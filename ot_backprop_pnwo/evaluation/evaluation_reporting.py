
import dataclasses
from typing import List
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

from ot_backprop_pnwo.evaluation.evaluation_param import ConvergenceConfig, TwoPhaseRunUniqueIdentifyingConfig
from ot_backprop_pnwo.optimization.emsc_loss_type import EMSCLossType
from ot_backprop_pnwo.optimization.ot_wo_two_phase import OT_WO_Result, OT_WO_Two_Phase
from ot_backprop_pnwo.spn.spn_wrapper import SPNWrapper
from ot_backprop_pnwo.util.my_util import flatten_dict

class ExistingInitialRunRepo():
    
    def __init__(self, l_path_log_results: List[Path]) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing the Initally Existing Run Repository")
        self._df_existing_runs = None

        dummy_key_config = TwoPhaseRunUniqueIdentifyingConfig(EMSCLossType.PEMSC, "", "", 0, 0, False, 0, False, convergence_config=ConvergenceConfig(0, 0, 0))
        set_required_columns = set(EvaluationReporterTwoPhase._get_flattened_key_param_dict(dummy_key_config).keys())
        self._key_order = list(set_required_columns)

        # Load Descriptions
        l_df_run_info = []
        for path_log_result in l_path_log_results:
            for path_info_file in path_log_result.glob(EvaluationReporterTwoPhase.run_info_file_pattern.format(RUNID='*')):
                with open(path_info_file) as f:
                    data = json.load(f)

                df = pd.json_normalize(data[EvaluationReporterTwoPhase.run_info_root_element])

                # Check if required  columns exists
                if set_required_columns.issubset(set(df.columns)):
                    l_df_run_info.append(df)
                else:
                    self.logger.error(f"Could not fine all required key columns in {str(path_info_file)}")

        if len(l_df_run_info) > 0:
            self._df_existing_runs = pd.concat(l_df_run_info, axis=0)
            # Drop all that are not needed
            set_existing_columns = set(df.columns)
            self._df_existing_runs = self._df_existing_runs.drop(
                list(set_existing_columns 
                     - (set_required_columns.union(
                         set((EvaluationReporterTwoPhase.column_name_id, EvaluationReporterTwoPhase.column_name_phase_two_exec))) )), axis=1)
            # Index
            self._df_existing_runs = self._df_existing_runs.set_index(self._key_order)
            self.logger.info(f"There are {len(self._df_existing_runs.index)} existing runs")
        else:
            self.logger.info("There are no existing runs")

    def get_next_free_id(self) -> int:
        if self._df_existing_runs is not None:
            # Maximum id
            return self._df_existing_runs[EvaluationReporterTwoPhase.column_name_id].astype(int).max() + 1
        else:
            return 0

    def is_run_required(self, key_run_param: TwoPhaseRunUniqueIdentifyingConfig) -> bool:
        if self._df_existing_runs is None:
            return True

        key_query = self._get_key_query(key_run_param)
        if self._df_existing_runs.index.isin([key_query]).any():
            return False
        else:
            return True
        
    def is_phase_two_executed_in_run(self, key_run_param: TwoPhaseRunUniqueIdentifyingConfig) -> bool:
        key_query = self._get_key_query(key_run_param)
        return self._df_existing_runs.loc[key_query, EvaluationReporterTwoPhase.column_name_phase_two_exec]

    def _get_key_query(self, key_run_param: TwoPhaseRunUniqueIdentifyingConfig):
        key_data_flattened = EvaluationReporterTwoPhase._get_flattened_key_param_dict(key_run_param)
        key_query = tuple(key_data_flattened[entry_name] for entry_name in self._key_order)
        return key_query


class EvaluationReporterTwoPhase():
    # Note. Could make this class generic (on key)
    column_name_id = 'idRun'
    column_name_phase_two_exec = 'phaseTwoExecuted'
    run_info_file_pattern = 'run-info-{RUNID}.json'
    run_info_root_element = 'runInfo'

    def __init__(self, existing_run_repo: ExistingInitialRunRepo, global_next_free_id, 
                 path_log_results: Path, lock=None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self._path_log_results = path_log_results
        self._global_next_free_id = global_next_free_id
        self._existing_run_repo = existing_run_repo
        self._lock = lock

    @property
    def path_log_results(self) -> Path:
        return self._path_log_results

    def is_run_required(self, key_run_param: TwoPhaseRunUniqueIdentifyingConfig) -> bool:
        return self._existing_run_repo.is_run_required(key_run_param)

    def is_phase_two_executed_in_run(self, key_run_param: TwoPhaseRunUniqueIdentifyingConfig) -> bool:
        return self._existing_run_repo.is_phase_two_executed_in_run(key_run_param)
        
    def report(self, key_param: TwoPhaseRunUniqueIdentifyingConfig, 
               spn_container: SPNWrapper, 
               ot_wo_result: OT_WO_Result, 
               spn_wo_alg: OT_WO_Two_Phase):

        if self._lock is not None:
            with self._lock:
                id_this_run = self._global_next_free_id.value
                self._global_next_free_id.value += 1
        else:
            id_this_run = self._global_next_free_id.value
            self._global_next_free_id.value += 1

        # Create Entry
        entry = EvaluationReporterTwoPhase._get_flattened_key_param_dict(key_param)
        entry.update(
            {
                EvaluationReporterTwoPhase.column_name_id: id_this_run,
                EvaluationReporterTwoPhase.column_name_phase_two_exec: spn_wo_alg.run_phase_two, # becomes automatically disabled if not needed
                'sizeLogPhaseOne': len(spn_wo_alg.data_phase_1.pseudo_log_lang),
                'sizeSPNLanguagePhaseOne': len(spn_wo_alg.data_phase_1.pseudo_spn_lang),
                'nbrSPNPathsPhaseOne': len(spn_wo_alg.data_phase_1.spn_paths),
                'timeTrainingP1': ot_wo_result.training_time_phase_one,
                'timeTrainingP2': ot_wo_result.training_time_phase_two,
                'timeFull': ot_wo_result.full_time,
                'is_path_most_likely_phase_one': spn_wo_alg.data_phase_1.is_path_most_likely,
                'nbr_iterations_phase_1': len(ot_wo_result.error_series_phase_one),
                'nbr_iterations_phase_2': 0 if ot_wo_result.error_series_phase_two is None else len(ot_wo_result.error_series_phase_two),
                'errorInitialP1': ot_wo_result.error_series_phase_one[0],
                'errorFinalP1': ot_wo_result.error_series_phase_one[-1],
                'errorInitialP2': -1 if ot_wo_result.error_series_phase_two is None else ot_wo_result.error_series_phase_two[0],
                'errorFinalP2': -1 if ot_wo_result.error_series_phase_two is None else ot_wo_result.error_series_phase_two[-1]
            }
        )
        ########################################
        # Run Info
        ########################################

        path_result_info_file = self.path_log_results / EvaluationReporterTwoPhase.run_info_file_pattern.format(RUNID=id_this_run)
        file_data = dict()
        file_data[EvaluationReporterTwoPhase.run_info_root_element] = entry
        with open(path_result_info_file, 'w+') as f:
            json.dump(file_data, f, indent=4)

        ########################################
        # Save GSPN and Error Series
        ########################################
        spn_container.update_transition_weights(np.abs(ot_wo_result.spn_weights))

        path_spn = self.path_log_results.joinpath(f"{key_param.name_spn}-{id_this_run}.pnml" )
        spn_container.export_to_file(path_spn)
        path_error_p1 = self.path_log_results.joinpath(f"error-series-{id_this_run}-p1.csv")
        np.savetxt(path_error_p1, ot_wo_result.error_series_phase_one, delimiter=",")
        if ot_wo_result.error_series_phase_two is not None:
            path_error_p2 = self.path_log_results.joinpath(f"error-series-{id_this_run}-p2.csv")
            np.savetxt(path_error_p2, ot_wo_result.error_series_phase_two, delimiter=",")

    @staticmethod
    def _get_flattened_key_param_dict(key_param: TwoPhaseRunUniqueIdentifyingConfig) -> dict:
        return flatten_dict(dataclasses.asdict(key_param))


if __name__ == '__main__':

    path_results = Path('ot-wo-evaluation/results-two-phase-explog-long')
    # Log - Petri net folder
    l_log_pn_pairs = (
        ('RTFM', 'logs/road_fines/road_fines.xes', 'models/road_fines/hot-start', True),
        ('RTFM', 'logs/road_fines/road_fines.xes', 'models/road_fines/cold-start', False),
        ('Sepsis', 'logs/sepsis/sepsis.xes', 'models/sepsis/hot-start', True),
        ('Sepsis', 'logs/sepsis/sepsis.xes', 'models/sepsis/cold-start', False),
        ('BPIC18Ref', 'logs/bpic-2018-reference/bpic-2018-reference.xes', 'models/bpic-2018-reference/hot-start', True),
        ('BPIC18Ref', 'logs/bpic-2018-reference/bpic-2018-reference.xes', 'models/bpic-2018-reference/cold-start', False),
    )
    log_names = set(log_name for (log_name, _, _, _) in l_log_pn_pairs)
    l_path_log_results = [path_results.joinpath(log_name) for log_name in log_names]
    initially_exiting_runs = ExistingInitialRunRepo(l_path_log_results)

    print(initially_exiting_runs._df_existing_runs)
    initially_exiting_runs._df_existing_runs.to_pickle('ot-wo-evaluation/results-two-phase-explog-long/eval-reporter-df.pkl')
    
    print(f'Next free id: {initially_exiting_runs.get_next_free_id()}')
        
