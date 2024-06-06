import json
import logging.config
from pathlib import Path
import unittest
import tempfile

import yaml

from ot_backprop_pnwo.evaluation import evaluation_two_phase


class TestPhaseTwoEvaluation(unittest.TestCase):
    
    TEST_DIR_TASKS = '/home/tobias/ot-wo-evaluation/data/'
    TEST_LOG_PN = (
        ('BPIC18Ref', 'logs/bpic-2018-reference/bpic-2018-reference.xes', 'models/bpic-2018-reference/hot-start', True),
        #('RTFM', 'logs/road_fines/road_fines.xes', 'models/road_fines/cold-start', False),
    )
    TEST_REPETITIONS = 1

    def test_evaluation_full(self):
        evaluation_two_phase.TEST_MODE = True
        with tempfile.TemporaryDirectory() as tmp_result_dir_name:
            paths_tasks = Path(TestPhaseTwoEvaluation.TEST_DIR_TASKS)
            paths_results = Path(tmp_result_dir_name)

            TEST_OT_SIZES = [(40, 40)]
            evaluation_two_phase.main(l_log_pn_pairs=TestPhaseTwoEvaluation.TEST_LOG_PN,
                                      path_tasks=paths_tasks, path_results=paths_results, 
                                      ot_sizes=TEST_OT_SIZES,
                                      repetitions=TestPhaseTwoEvaluation.TEST_REPETITIONS)

            nbr_results_invoke_1 = len(list(paths_results.glob('**/run-info-*.json')))

            print('==================== TEST REPEATED RUN ====================')

            TEST_OT_SIZES = [(40, 40), (10, 40)]
            # Second run should result in skipping
            evaluation_two_phase.main(l_log_pn_pairs=TestPhaseTwoEvaluation.TEST_LOG_PN,
                                      path_tasks=paths_tasks, path_results=paths_results, 
                                      ot_sizes=TEST_OT_SIZES,
                                      repetitions=TestPhaseTwoEvaluation.TEST_REPETITIONS, override=False)

            nbr_results_invoke_2 = len(list(paths_results.glob('**/*.json')))
            
            # Print
            for i, res_file in enumerate(paths_results.glob('**/*.json')):
                with open(res_file) as f_res:
                    data = json.load(f_res)
                    print(f'File {i + 1}')
                    print(data)

            self.assertEqual(nbr_results_invoke_2, 2 * nbr_results_invoke_1)


if __name__ == '__main__':
    # Load the config file
    with open('ot_backprop_pnwo/logger-config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())

    # Configure the logging module with the config file
    logging.config.dictConfig(config)

    unittest.main()