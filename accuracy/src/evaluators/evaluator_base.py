"""base evaluator"""
from evaluators.dataprocessor_repository import DATAPROCESSORREPOSITORY

class EvaluatorBase:
    """EvaluatorBase"""

    def __init__(self, config):
        self._input_dir = config.input_dir
        self._output_dir = config.output_dir
        self._dataset_filename = config.dataset_filename
        self._gkwargs = config.gkwargs
        DataProcessor = DATAPROCESSORREPOSITORY.get_data_processor(config.dataprocessor_name)
        self._data_processor = DataProcessor(config)

    def preprocess(self):
        pass

    def sample(self):
        pass

    def eval(self):
        pass

    def postprocess(self):
        pass

    def exec_pipeline(self):
        pass
