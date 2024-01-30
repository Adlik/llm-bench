"""evaluator"""
import os
import json

from utils.evaluator_util import Config
from evaluators.repository import EVALUATORREPOSITORY


def multiple_rounds(EVALUATOR, config):
    output_file = config.output_dir+"/result.json"
    extracted_data = {
        'language': "",
        'runtime': config.runtime_type,
        'result_data': {}
    }
    sum_values = 0
    output_path = config.output_dir
    for i in range(config.eval_times):
        config = config._replace(output_dir=f'{output_path}/output_{i+1}')
        os.makedirs(config.output_dir, exist_ok=True)
        evaluator = EVALUATOR(config)
        language, result = evaluator.exec_pipeline()
        extracted_data['language'] = language
        extracted_data['result_data'][f'round_{i+1}'] = result
        sum_values += float(result)
        extracted_data['result_data']['average'] = sum_values / (i+1)
        average_value = extracted_data['result_data']['average']

        del extracted_data['result_data']['average']
        extracted_data['result_data']['average'] = average_value
        with open(output_file, 'w', encoding="utf-8") as outfile:
            json.dump(extracted_data, outfile, indent=4)
    file_path = f"{output_path}/finish.txt"
    with open(file_path, 'w', encoding="utf-8") as fid:
        fid.write('finish')


def evaluate_from_json(value):
    config = Config.from_json(value)

    EVALUATOR = EVALUATORREPOSITORY.get_evaluator(config.eval_type)
    multiple_rounds(EVALUATOR, config)


def evaluate_from_env():
    config = Config.from_env(os.environ.copy())

    EVALUATOR = EVALUATORREPOSITORY.get_evaluator(config.eval_type)
    multiple_rounds(EVALUATOR, config)
