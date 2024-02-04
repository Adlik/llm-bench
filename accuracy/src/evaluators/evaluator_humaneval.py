"""evaluator_humaneval"""
import os
import time
import json
import logging
import sys

from human_eval.evaluation import (
    evaluate_functional_correctness as HUMANEVAL_PYTHON_EVALUATE,
)
import evaluators.repository as EVALUATORS_REPOSITORY
from .evaluator_base import EvaluatorBase
from .llmgpt import LLMgptModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EVALUATION")


@EVALUATORS_REPOSITORY.EVALUATORREPOSITORY.register_evaluator("HumanEval")
class EvaluatorHumaneval(EvaluatorBase):
    """EvaluatorHumaneval"""

    def __init__(self, config):
        super().__init__(config)
        self._eval_type = config.eval_type
        self._language = config.language
        self._check_file_path()
        self.data = {
            "sample_file": os.path.join(
                self._output_dir,
                f"{self._eval_type}_{self._language}_results.jsonl",
            ),
        }
        self._dataset_name = self._dataset_filename[:self._dataset_filename.find(".jsonl")]

    def _check_file_path(self):
        self._problem_file = os.path.join(self._input_dir, self._dataset_filename)
        if not os.path.exists(self._problem_file):
            logger.error(f"ERROR: File {self._problem_file} is not exist!")
            sys.exit(1)
        self._single_res_path = os.path.join(
            self._output_dir, f"{self._eval_type}_jsons"
        )
        os.makedirs(self._single_res_path, exist_ok=True)

    def _save_simple_result(self, json_data, timestamp, answers, idx):
        json_data["response"] = {
            "timestamp_start": timestamp[0],
            "timestamp_end": timestamp[1],
            "gkwargs": self._gkwargs,
            "answers": answers,
        }
        output_json_path = f"{self._single_res_path}/{idx}.json"
        with open(output_json_path, "w", encoding="utf-8") as fid:
            fid.write(json.dumps(json_data, indent=4, ensure_ascii=False))

    def infer(self, fid):
        total_answer = []
        timestmp_start = time.time()
        self._gkwargs = json.loads(self._gkwargs)
        llmgpt_model = LLMgptModel()
        logger.info(f"---------------------------{self._eval_type} infer start")
        for line in fid:
            json_data = json.loads(line)
            idx = json_data["task_id"]
            idx = idx.replace('/', '_')
            logger.info(f"--------------------------- infer {idx}")
            input_data = {
                'eval_type': self._eval_type,
                'language': self._language,
                'json_data': json_data
            }
            question = self._data_processor.process_input(input_data)
            logger.info(f"--------------------------- question\n {question}")
            answers_rm_question, timestamp_start, timestamp_end = llmgpt_model.chat(question, self._gkwargs)
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
                    "task_id": json_data["task_id"],
                    "completion": answer,
                }
                for answer in json_data["response"]["answers"]
            ]
            total_answer += data_answers
        timestmp_end = time.time()
        logger.info(f"Inference time:{(timestmp_end - timestmp_start) / 60:.2f} min")
        return total_answer

    def sample(self):
        total_answer = []
        language = "Python" if self._language == "" else self._language
        self.data['sample_file'] = os.path.join(
            self._output_dir,
            f"{self._eval_type}_{language}_results.jsonl",
        )
        with open(self._problem_file, "r", encoding="utf-8") as fid:
            total_answer = self.infer(fid)

        with open(self.data['sample_file'], "w", encoding="utf-8") as res:
            res.write("\n".join([json.dumps(data, ensure_ascii=False) for data in total_answer]))

    def extract_evaluation_results(self, results):

        if self._dataset_name.endswith("_CN"):
            natural_language = "cn"
        else:
            natural_language = "en"
        lang_name = f'{self._language}_{natural_language}'

        return lang_name, results['pass@1']

    def eval(self):
        results = HUMANEVAL_PYTHON_EVALUATE(
            self.data['sample_file'], problem_file=self._problem_file
        )
        logger.info(f"\n\n---------------------------{self._eval_type} infer results")
        logger.info(results)
        return results

    def exec_pipeline(self):
        self.sample()
        results = self.eval()
        lang_name, result = self.extract_evaluation_results(results)
        return lang_name, result
