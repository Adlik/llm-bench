# Copyright 2023 ZTE corporation. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
'''
This is a model class for LLMgpt
'''

import os
import sys
import json
import time
import logging
from typing import Dict, Tuple, List, Union

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


TokenSequence = Union[List[int], torch.Tensor]


class LLMgptModel:
    ''' LLM GPT Model
    '''

    def __init__(self):
        self.use_gpu = int(os.environ.get('LLM_GPT_USE_GPU_NUM', 1))
        self.use_quantize_int = int(
            os.environ.get('LLM_GPT_USE_QUANTIZE_INT', 0))
        model_dir = os.environ.get('LLM_GPT_MODEL_DIR', '/LLM/model')
        self.log_dir = os.environ.get('LLM_GPT_LOG_DIR', None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=bool(int(os.environ.get('LLM_GPT_TOKENIZER_USE_FAST',
                                             0))))
        self._set_special_tokenizer_token()
        self.rolling_max_length = int(
            os.environ.get('LLM_GPT_ROLLING_MAX_LENGTH', 1024))

        if self.use_gpu == 1 and self.use_quantize_int == 0:
            self.torch_model = AutoModelForCausalLM.from_pretrained(
                model_dir).half().cuda().eval()
            self._set_special_model_token()
        elif self.use_gpu >= 1:
            self.torch_model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                load_in_8bit=(self.use_quantize_int == 8),
                torch_dtype=torch.float16,  # pylint: disable=no-member
                device_map='auto')
            self.torch_model.eval()
            self._set_special_model_token()
            if torch.__version__ >= '2' and sys.platform != 'win32':
                self.torch_model = torch.compile(self.torch_model)
        elif self.use_gpu == 0 and self.use_quantize_int == 0:
            self.torch_model = AutoModelForCausalLM.from_pretrained(
                model_dir, torchscript=True).eval()
        else:
            raise ValueError(
                'Invalid LLM_GPT_USE_GPU_NUM and LLM_GPT_USE_QUANTIZE_INT')

    def _set_special_tokenizer_token(self):
        tokenizer_pad_token_id_str = os.environ.get('LLM_GPT_TOKENIZER_PAD_ID',
                                                    None)
        if tokenizer_pad_token_id_str is not None:
            self.tokenizer.pad_token_id = int(tokenizer_pad_token_id_str)

    def _set_special_model_token(self):
        model_pad_token_id_str = os.environ.get('LLM_GPT_MODEL_PAD_ID', None)
        model_bos_token_id_str = os.environ.get('LLM_GPT_MODEL_BOS_ID', None)
        model_eos_token_id_str = os.environ.get('LLM_GPT_MODEL_EOS_ID', None)

        if model_pad_token_id_str is not None:
            self.torch_model.config.pad_token_id = int(model_pad_token_id_str)
        if model_bos_token_id_str is not None:
            self.torch_model.config.bos_token_id = int(model_bos_token_id_str)
        if model_eos_token_id_str is not None:
            self.torch_model.config.eos_token_id = int(model_eos_token_id_str)

    def _save_log(self,
                  json_data: Dict,
                  file_name_prefix: str = 'chat-server-') -> None:
        if self.log_dir is None:
            return
        if not os.path.exists(self.log_dir):
            return
        file_name = f'{self.log_dir}/{file_name_prefix}{json_data["timestamp_start"]}.json'
        with open(file_name, 'a', encoding='utf-8') as fid:
            fid.write(json.dumps(json_data))

    def chat(self, question: str, gkwargs: Dict) -> Tuple:
        ''' give answers for question with gkwargs
        '''
        timestamp_start = time.time()
        inputs = self.tokenizer(
            question,
            max_length=gkwargs.get('max_length_input', 1024),
            truncation=bool(gkwargs.get('truncation', False)),
            return_tensors=gkwargs.get('return_tensors', 'pt'))

        input_ids_list = inputs.input_ids.tolist()[0]

        if self.use_gpu:
            input_ids = inputs.input_ids.to('cuda')
        else:
            input_ids = inputs.input_ids
        with torch.no_grad():
            generation_output = self.torch_model.generate(
                input_ids=input_ids,
                max_length=gkwargs.get('max_length_output', 1024),
                num_beams=gkwargs.get('num_beams', 1),
                num_return_sequences=gkwargs.get('num_return_sequences', 1),
                no_repeat_ngram_size=gkwargs.get('no_repeat_ngram_size', 0),
                repetition_penalty=gkwargs.get('repetition_penalty', 1.0),
                temperature=gkwargs.get('temperature', 0),
                top_k=gkwargs.get('top_k', 50),
                top_p=gkwargs.get('top_p', 1.0),
                do_sample=bool(gkwargs.get('do_sample', False)),
                remove_invalid_values=bool(
                    gkwargs.get('remove_invalid_values', True)))

        output_ids_list = [item.tolist() for item in generation_output]

        timestamp_gend = time.time()
        json_data_gend = {
            'timestamp_start': timestamp_start,
            'timestamp_gend': timestamp_gend,
            'question': question,
            'input_ids_list': input_ids_list,
            'output_ids_list': output_ids_list,
            'gkwargs': gkwargs
        }
        self._save_log(json_data_gend, 'chat-tmp-server-')

        answers = [
            self.tokenizer.decode(item,
                                  skip_special_tokens=bool(
                                      gkwargs.get('skip_special_tokens',
                                                  True)))
            for item in generation_output
        ]

        answers_rm_question = [
            self.tokenizer.decode(item[len(input_ids_list):],
                                  skip_special_tokens=bool(
                                      gkwargs.get('skip_special_tokens',
                                                  False)))
            for item in output_ids_list
        ]

        timestamp_end = time.time()
        json_data_end = {
            'timestamp_start': timestamp_start,
            'timestamp_gend': timestamp_gend,
            'timestamp_end': timestamp_end,
            'question': question,
            'answers': answers,
            'answers_rm_question': answers_rm_question,
            'question_tokens_str': json.dumps(input_ids_list),
            'answers_tokens_str': [json.dumps(item.tolist()) for item in generation_output],
            'gkwargs': gkwargs
        }

        self._save_log(json_data_end)
        logging.debug(json_data_end)

        return answers, answers_rm_question, timestamp_start, timestamp_end

    def _model_call(self, inputs) -> TokenSequence:
        return self.torch_model(inputs)[0]

    def _encode(self, text: str) -> TokenSequence:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _make_disjoint_window(self, group):
        a, b, c = group
        b = b[-c:]
        return a[: len(a) - (len(b) - 1)], b
