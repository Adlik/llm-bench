"""DataProcessorBase"""
from typing import Dict, List, Union
class DataProcessorBase:
    """DataProcessorBase"""
    def __init__(self, config):
        self._dataprocessor_name = config.dataprocessor_name
        self._promt_templates = {}

    def process_input(self, input_data : Dict[str, Union[str, Dict[str, str]]]) -> str:
        """Process input data and generate a formatted question.
        Args:
            input_data (Dict[str, Union[str, Dict[str, str]]]): A dictionary containing the following keys:
                - 'eval_type' (str): The type of evaluation (e.g., HumanEval, BabelCode).
                - 'language' (str): The programming language for which to generate the question.
                - 'json_data' (Dict[str, str]): JSON data containing additional input information.

        Returns:
            str: The formatted question.
        """
        pass

    def process_output(self, output_data: str) -> Dict[str, List[str]]:
        """Process output data and reformat it if needed.
        Args:
            output_data (Dict[str, Union[str, Dict[str, str]]]): A dictionary containing the following keys:
                - 'eval_type' (str): The type of evaluation (e.g., HumanEval, BabelCode).
                - 'language' (str): The programming language of the output.
                - 'json_data' (Dict[str, List[str]]): JSON data containing output information by model.

        Returns:
            json_data Dict[str, List[str]]: Processed JSON data.
        """
        pass

    def add_template(self, language: str) -> Dict[str, str]:
        """Add template for the specified programming language.

        Args:
            language (str): The programming language for which to add a template.

        Returns:
            Dict[str, str]: A dictionary containing templates for different languages.
        """
        pass
