""" Evaluator repo"""

from typing import Dict

class EvaluatorRepository:
    """EvaluatorRepository"""
    def __init__(self):
        self._evaluators: Dict[str, type] = {}

    def register_evaluator(self, eval_type):
        def _register(evaluator: type):
            self._evaluators[eval_type] = evaluator

            return evaluator

        return _register

    def get_evaluator(self, eval_type):
        return self._evaluators[eval_type]


EVALUATORREPOSITORY = EvaluatorRepository()
