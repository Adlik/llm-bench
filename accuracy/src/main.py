"""
    Copyright 2019 ZTE corporation. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""
import sys
import json
import logging
import argparse
from datetime import datetime

import evaluator


def setup_logger(log_file):
    logger = logging.getLogger("EVALUATION")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')

    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def main():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = f"/output/evaluation_{current_time}.log"
    logger = setup_logger(log_file)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--config-source',
                        default='env',
                        choices=['env', 'json'],
                        required=False,
                        help='evaluate of benchmark config, can be env or json')

    parser.add_argument('-p', '--json-file', type=str, required=False, default=None, help='path of json file')

    args = parser.parse_args()

    if args.config_source == 'json':
        logger.info("Start parsing parameters from json file.")
        with open(args.json_file, encoding="utf-8") as json_file:
            evaluator.evaluate_from_json(json.load(json_file))
    else:
        logger.info("Start parsing parameters from environment variables.")
        evaluator.evaluate_from_env()


if __name__ == '__main__':
    main()
