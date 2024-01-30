""" Data_processor repo"""

import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EVALUATION")


class DataProcessorRepository:
    """DataProcessorRepository"""

    def __init__(self):
        self._data_processors: Dict[str, type] = {}

    def register_dataprocessor(self, dataprocessor_name):
        def _register(data_processor: type):
            self._data_processors[dataprocessor_name] = data_processor
            return data_processor

        return _register

    def get_data_processor(self, dataprocessor_name):
        try:
            data_processor = self._data_processors[dataprocessor_name]
            logger.info(f"Get DataProcessor {data_processor}")
        except KeyError as error:
            logger.error(f"{dataprocessor_name} DataProcessor not registered. Cannot use it")
            raise KeyError from error
        return data_processor


DATAPROCESSORREPOSITORY = DataProcessorRepository()
