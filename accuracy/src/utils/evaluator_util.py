"""evaluator_util"""
import json
import logging
from typing import Any, Mapping, NamedTuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EVALUATION")


class Config(NamedTuple):
    """Config"""

    input_dir: Optional[str]
    output_dir: Optional[str]
    runtime_type: Optional[str]
    dataset_filename: Optional[str]
    eval_type: Optional[str]
    language: Optional[str]
    dataprocessor_name: Optional[str]
    gkwargs: Optional[str]
    eval_times: Optional[int]

    @staticmethod
    def from_json(value: Mapping[str, Any]) -> "Config":
        return Config(
            input_dir=value.get("input_dir", "data"),
            output_dir=value.get("output_dir", "/output"),
            runtime_type=value.get("runtime_type", "hft"),
            dataset_filename=value.get("dataset_filename", "HumanEval.jsonl"),
            eval_type=value.get("eval_type", "HumanEval"),
            language=value.get("language", "Java"),
            dataprocessor_name=value.get("dataprocessor_name", "codellama-34b-base-hft"),
            gkwargs=value.get("gkwargs", "{}"),
            eval_times=value.get("eval_times", 1)
        )

    @staticmethod
    def from_env(env: Mapping[str, str]) -> "Config":
        gkwargs = get_keyword_data("GKWARGS_", env)
        ekwargs = get_keyword_data("EKWARGS_", env)
        return Config(
            input_dir=env.get("INPUT_DIR", "data"),
            output_dir=env.get("OUTPUT_DIR", "/output"),
            runtime_type=gkwargs.get("runtime_type", "hft"),
            gkwargs=json.dumps(gkwargs),
            dataset_filename=ekwargs.get("dataset_filename", "HumanEval.jsonl"),
            eval_type=ekwargs.get("eval_type", "HumanEval"),
            language=ekwargs.get("language", "Java"),
            dataprocessor_name=ekwargs.get("dataprocessor_name", "codellama-34b-base-hft"),
            eval_times=int(ekwargs.get("eval_times", "1"))
        )


def get_keyword_data(prefix, env):
    data_tmp = ""
    i = 0
    while True:
        keyword = f"{prefix}{i}"
        tmp = env.get(keyword, 'null')
        if tmp == 'null':
            break
        data_tmp += tmp
        i += 1
    kwargs = "{}" if data_tmp == "" else data_tmp
    kwargs = json.loads(kwargs)
    return kwargs
