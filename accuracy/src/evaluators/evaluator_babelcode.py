"""evaluator_babelcode"""
import re
import os
import sys
import time
import json
from pathlib import Path
import logging
import shutil
import subprocess

import evaluators.repository as EVALUATORS_REPOSITORY
from .evaluator_base import EvaluatorBase
from .llmgpt import LLMgptModel
from convert_dataset import convert_dataset
from generate_test_code import generate_problem_code_main

sys.path.append('/babelcode')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EVALUATION")


@EVALUATORS_REPOSITORY.EVALUATORREPOSITORY.register_evaluator("BabelCode")
class Evaluatorbabelcode(EvaluatorBase):
    """Evaluatorbabelcode"""

    def __init__(self, config):
        super().__init__(config)
        self._eval_type = config.eval_type
        self._language = config.language
        self._check_file_path()
        self.data = {
            "dataset_name": self._dataset_filename[:self._dataset_filename.find(".jsonl")],
            "bablcode_dataset_name": "",
            }
        if self._eval_type == "BabelCode":
            if self._dataset_filename == "BabelCode_HumanEval.jsonl":
                self.data["bablcode_dataset_name"] = "human_eval_en"
            if self._dataset_filename == "BabelCode_HumanEval_CN.jsonl":
                self.data["bablcode_dataset_name"] = "human_eval_cn"

    def _check_file_path(self):
        self._problem_file = os.path.join(self._input_dir, self._dataset_filename)
        if not os.path.exists(self._problem_file):
            logger.error(f"ERROR: File {self._problem_file} is not exist!")
            sys.exit(1)

        self._single_res_path = os.path.join(
            self._output_dir, f"{self._eval_type}_jsons"
        )
        os.makedirs(self._single_res_path, exist_ok=True)

    def preprocess(self):
        dataset_name = self.data['bablcode_dataset_name']
        logger.info(f"================dataset_name {dataset_name}")
        convert_dataset(
            dataset_name=self.data['bablcode_dataset_name'],
            input_path=Path(self._problem_file),
            disable_fixes=False,
            debug=False,
            debug_question=None
        )

        try:
            os.makedirs('configs/generation')
            logger.info("Folder 'configs/generation' has been created successfully")
        except OSError:
            logger.info("Folder 'configs/generation' already exists")

        try:
            shutil.copyfile('/babelcode/configs/generation/base.gin', 'configs/generation/base.gin')
            logger.info("File 'generation/base.gin' copied successfully")
        except shutil.Error:
            logger.info("Copy file 'generation/base.gin' failed")

        generate_problem_code_main(
            gin_path="/babelcode/configs/generate_code.gin",
            input_path=Path(f"/babelcode/data/parsed_datasets/{self.data['bablcode_dataset_name']}.jsonl"),
            output_path=Path(f"/babelcode/data/problem_code/{self.data['bablcode_dataset_name']}"),
            debug_lang=None,
            debug=False
        )

        try:
            shutil.rmtree('configs')
            logger.info("Folder 'configs' and its content removed")
        except shutil.Error:
            logger.info("Folder 'configs' not deleted")

        input_file = f"/babelcode/data/problem_code/{self.data['bablcode_dataset_name']}/prompt_info.jsonl"
        output_file = f"{self._output_dir}/babelcode_{self._language}.jsonl"
        target_field = "language"
        target_value = self._language
        self._filter_json_objects(input_file, output_file, target_field, target_value)

    def _filter_json_objects(self, input_file, output_file, target_field, target_value):
        with open(input_file, "r", encoding="utf-8") as infile, open(
            output_file, "w", encoding="utf-8"
        ) as outfile:
            for line in infile:
                try:
                    json_object = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    target_field in json_object
                    and json_object[target_field] == target_value
                ):
                    json.dump(json_object, outfile)
                    outfile.write("\n")

    def _save_simple_result(self, json_data, timestamp, answers, idx):
        json_data["response"] = {
            "timestamp_start": timestamp[0],
            "timestamp_end": timestamp[1],
            "gkwargs": self._gkwargs,
            "answers": answers,
        }
        output_json_path = f"{self._single_res_path}/{self._eval_type}_{idx}.json"
        with open(output_json_path, "w", encoding="utf-8") as fid:
            fid.write(json.dumps(json_data, indent=4))

    def infer(self, fid):
        total_answer = []
        timestmp_start = time.time()
        logger.info(
            f"---------------------------{self._eval_type} {self._language} infer start"
        )
        self._gkwargs = json.loads(self._gkwargs)
        llmgpt_model = LLMgptModel()
        for idx, line in enumerate(fid):
            json_data = json.loads(line)
            logger.info(f"--------------------------- infer {idx}")
            input_data = {
                'eval_type': self._eval_type,
                'language': self._language,
                'json_data': json_data
            }
            question = self._data_processor.process_input(input_data)
            logger.info(f"--------------------------- question\n{question}")
            answers, answers_rm_question, timestamp_start, timestamp_end = llmgpt_model.chat(question, self._gkwargs)
            self._save_simple_result(
                json_data, [timestamp_start, timestamp_end], answers_rm_question, idx
            )
            output_data = {
                'eval_type': self._eval_type,
                'language': self._language,
                'json_data': json_data
            }
            json_data = self._data_processor.process_output(output_data)
            data_answers = [
                {
                    "qid": json_data["qid"],
                    "language": json_data["language"],
                    "code": answer,
                }
                for answer in json_data["response"]["answers"]
            ]
            total_answer += data_answers
        timestmp_end = time.time()
        logger.info(f"Inference time: {(timestmp_end - timestmp_start) / 60:.2f} min")
        return total_answer

    def sample(self):
        input_json_path = os.path.join(
            self._output_dir, f"babelcode_{self._language}.jsonl"
        )
        with open(input_json_path, "r", encoding="utf-8") as fid:
            total_answer = self.infer(fid)

        result_file = (
            f"{self._output_dir}/{self._eval_type}_{self._language}_result.jsonl"
        )
        with open(result_file, "w", encoding="utf-8") as fid:
            fid.write("\n".join([json.dumps(data) for data in total_answer]))

    def extract_evaluation_results(self):
        value_pattern = r'estimate_pass@1\s+=\s+([\d.]+)'
        with open(f"{self._output_dir}/tutorial_{self._language}/logs/logs.INFO", 'r', encoding="utf-8") as file:
            lines = file.readlines()
            last_n_lines = lines[-6:]
            for line in last_n_lines:
                match = re.search(value_pattern, line)
                if match:
                    value = round(float(match.group(1)) / 100.0, 5)

        if self.data['dataset_name'].endswith("_CN"):
            natural_language = "cn"
        else:
            natural_language = "en"
        lang_name = f'{self._language}_{natural_language}'

        return lang_name, value

    def eval(self):
        command_evaluate = [
            "python3",
            "evaluate_predictions.py",
            "--gin_file=configs/validation.gin",
            f"--experiment_name=tutorial_{self._language}",
            f"--predictions={self._output_dir}/{self._eval_type}_{self._language}_result.jsonl",
            f"--output_path={self._output_dir}",
            f"--test_code=data/problem_code/{self.data['bablcode_dataset_name']}",
            f"--debug_dir={self._output_dir}",
            "--overwrite",
        ]
        subprocess.run(
            command_evaluate,
            check=True,
            cwd="/babelcode",
        )

    def exec_pipeline(self):
        self.preprocess()
        self.sample()
        self.eval()
        lang_name, result = self.extract_evaluation_results()
        return lang_name, result
