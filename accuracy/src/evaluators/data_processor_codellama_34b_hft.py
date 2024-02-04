"""CodellamaDataProcessorHft"""
# pylint: disable=bare-except,too-many-branches,too-many-statements
import logging
from evaluators import dataprocessor_repository
from .data_processor_base import DataProcessorBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EVALUATION")


@dataprocessor_repository.DATAPROCESSORREPOSITORY.register_dataprocessor(
    "codellama-34b-base-hft"
)
class CodellamaDataProcessorHft(DataProcessorBase):
    """CodellamaDataProcessorHft"""

    def add_tempalte(self, language):
        self._promt_templates[language] = (
            """Below is an instruction that describes a task. \
Write a response that appropriately completes the request.\n\n\n\
### Instruction:\nCreate a {} script for this problem:\n{}\n\n### Response:"""
        )
        return self._promt_templates

    def process_input(self, input_data):
        eval_type = input_data["eval_type"]
        language = input_data["language"]
        json_data = input_data["json_data"]
        _promt_templates = self.add_tempalte(language)
        if eval_type == "HumanEval":
            input_prompt = json_data["prompt"]
            question = _promt_templates[language].format(language, input_prompt)
        elif eval_type == "BabelCode":
            question = self._promt_templates[language].format(
                language, json_data["signature_with_docstring"]
            )
        return question

    def process_output(self, output_data):
        language = output_data["language"]
        json_data = output_data["json_data"]
        answers = []
        for origin_answer in json_data["response"]["answers"]:
            logger.info(f"--------------------------- origin_answer\n{origin_answer}")
            completion = origin_answer
            if "\r" in completion:
                completion = completion.replace("\r", "")
            if language == "Python":
                if '```python' in completion:
                    def_line = completion.index('```python')
                    completion = completion[def_line:].strip()
                    completion = completion.replace('```python', '')
                if '__name__ == \"__main__\"' in completion:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                if "# Example usage" in completion:
                    next_line = completion.index('# Example usage')
                    completion = completion[:next_line].strip()
                if "###" in completion:
                    next_line = completion.index("###")
                    completion = completion[:next_line].strip()
                if "</s>" in completion:
                    next_line = completion.index("</s>")
                    completion = completion[:next_line].strip()

            if language == "Java":
                if "class Solution {" in completion:
                    def_line = completion.index("class Solution {")
                    completion = completion[def_line:].strip()
                    if "```" in completion:
                        end_index = completion.find("```")
                        completion = completion[:end_index].strip()
                    if "###" in completion:
                        next_line = completion.index("###")
                        completion = completion[:next_line].strip()
                    if "</s>" in completion:
                        next_line = completion.index("</s>")
                        completion = completion[:next_line].strip()

                elif "public" in completion:
                    def_line = completion.index("public")
                    completion = completion[def_line:].strip()
                    if "```" in completion:
                        end_index = completion.find("```")
                        completion = completion[:end_index].strip()
                    if "###" in completion:
                        next_line = completion.index("###")
                        completion = completion[:next_line].strip()
                    if "</s>" in completion:
                        next_line = completion.index("</s>")
                        completion = completion[:next_line].strip()
                    completion = "class Solution {\n    " + completion + "\n}\n"

            if language == "Go":
                if json_data["signature"] in completion:
                    def_line = completion.index(json_data["signature"])
                    completion = completion[def_line:].strip()
                if 'func main()' in completion:
                    end_line = completion.index('func main()')
                    completion = completion[:end_line].strip()

            if language == "C++":
                if json_data["signature"] in completion:
                    def_line = completion.index(json_data["signature"])
                    completion = completion[def_line:].strip()
                if 'int main()' in completion:
                    end_line = completion.index('int main()')
                    completion = completion[:end_line].strip()

            if '```' in completion:
                next_line = completion.index('```')
                completion = completion[:next_line].strip()
            logger.info(f"--------------------------- completion\n{completion}")
            answers.append(completion)
            json_data["response"]["answers"] = answers
        return json_data
