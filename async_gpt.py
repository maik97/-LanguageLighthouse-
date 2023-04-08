import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime

import httpx
import openai
from typing import List, Dict, Tuple, Optional


def flatten_list_basic(nested_list):
    flattened_list = []

    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list_basic(item))
        else:
            flattened_list.append(item)

    return flattened_list


def flatten_list(nested_list):
    flat_list = []
    index_map = []

    def _flatten_list(sublist, index_structure):
        if isinstance(sublist, list):
            for i, item in enumerate(sublist):
                _flatten_list(item, index_structure + [i])
        else:
            flat_list.append(sublist)
            index_map.append(index_structure)

    _flatten_list(nested_list, [])

    return flat_list, index_map


def unflatten_list(flat_list, index_map):
    def _create_empty_structure(index_map):
        structure = []
        for index in index_map:
            current_level = structure
            for i, idx in enumerate(index[:-1]):
                while len(current_level) <= idx:
                    current_level.append([])

                current_level = current_level[idx]

        return structure

    original_structure = _create_empty_structure(index_map)

    for value, index in zip(flat_list, index_map):
        current_level = original_structure
        for i, idx in enumerate(index[:-1]):
            current_level = current_level[idx]

        if len(current_level) <= index[-1]:
            current_level.append(value)
        else:
            current_level[index[-1]] = value

    return original_structure


def iterable_dict_generator(iterables_dict):
    if not iterables_dict:
        yield {}
    else:
        keys = list(iterables_dict.keys())
        n = len(iterables_dict[keys[0]])
        for i in range(n):
            subdict = {k: iterables_dict[k][i] for k in keys}
            yield subdict

def reverse_iterable_dict_generator(subdict_generator):
    result_dict = {}
    keys = None

    for subdict in subdict_generator:
        if keys is None:
            keys = list(subdict.keys())
            for key in keys:
                result_dict[key] = []

        for key in keys:
            result_dict[key].append(subdict[key])

    return result_dict

class ResponseProcessor:

    def __init__(self, eval_function, eval_function_kwargs=None, eval_iterables_dict=None, eval_valid_exceptions=None):
        self.failed_payloads = []
        self.failed_payloads_index = []
        self.eval_function = eval_function
        self.eval_function_kwargs = eval_function_kwargs or {}
        self.iterable_eval_kwargs = [subdict for subdict in iterable_dict_generator(eval_iterables_dict or {})]
        self.valid_exceptions = eval_valid_exceptions or (Exception,)

    def __call__(self, idx_map, responses, payloads):
        for idx, response, payload in zip(idx_map, responses, payloads):
            try:
                eval_kwargs = {}
                eval_kwargs.update(self.eval_function_kwargs.copy())
                eval_kwargs.update(self.iterable_eval_kwargs[idx].copy())
                response = self.eval_function(response=response, **eval_kwargs)
                yield idx, response
            except self.valid_exceptions as e:
                print('\033[35m\nResponse Evaluation Failed.\033[0m')
                print('\033[35mResponse:\n\033[0m', response)
                print('\033[35mException:\n\033[0m', e)
                print('\033[35mIndex:\n\033[0m', idx)
                print('\033[35mPayload:\n\033[0m', payload)
                print('\033[35meval_kwargs:\n\033[0m', eval_kwargs)
                self.failed_payloads.append(payload)
                self.failed_payloads_index.append(idx)

    def reset(self):
        failed_payloads, failed_payloads_index = self.failed_payloads, self.failed_payloads_index
        self.failed_payloads = []
        self.failed_payloads_index = []

        eval_iterables_dict = defaultdict(list)
        for idx in failed_payloads_index:
            for key, value in self.iterable_eval_kwargs[idx].items():
                eval_iterables_dict[key].append(value)

        return failed_payloads, failed_payloads_index, eval_iterables_dict


class AsyncChatGPT:
    """
    A class for generating chat completions using OpenAI's GPT language model.
    """

    def __init__(
            self,
            model: str = "gpt-3.5-turbo",
            completion_kwargs: Optional[Dict[str, any]] = None,
            log_dir: Optional[str] = None,
    ):
        """
        Initialize the ChatGPTInstance object.

        :param model: The name of the GPT model to use.
        :param completion_kwargs: A dictionary of keyword arguments to pass to the OpenAI ChatCompletion.create method.
        :param log_dir: The directory to log completion requests and responses to.
        """
        self.model = model
        self.completion_kwargs = completion_kwargs or {}
        self.log_dir = log_dir
        self.reset()

    def prompt_test(self, prompt=None, prompt_file=None, prompt_kwargs=None, messages=None, system_instruction=None, payload_kwargs=None, debug_prints=True):
        payload = self.make_payload(prompt=prompt, prompt_file=prompt_file, prompt_kwargs=prompt_kwargs, messages=messages, system_instruction=system_instruction, payload_kwargs=payload_kwargs)
        if debug_prints:
            print('\n\033[35mprompt\033[0m')
            print(payload['messages'][-1]['content'])
        response = self.get_responses(payloads=payload)
        if debug_prints:
            print('\n\033[35mresponse\033[0m')
            print(response)
        return response

    def reset(self):
        self.updated_messages = None
        self.token_count = 0

    def process_pre_prompt(self, pre_prompt):
        if not pre_prompt:
            return ''
        else:
            return pre_prompt + '\n\n'

    def log_completions(self, messages: List[Dict[str, str]], response_text: str, total_tokens: int) -> None:
        """
        Log the completion request and response to a JSON file.

        :param messages: A list of messages in the format returned by make_messages.
        :param response_text: The text of the completion response.
        :param total_tokens: The number of tokens used.
        :return: None
        """
        if not self.log_dir:
            return

        # Create a filename based on the class name and current date/time
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.__class__.__name__}_{timestamp}.json"

        # Create a dictionary with the completion request and response data
        data = {
            "timestamp": timestamp,
            "messages": messages,
            "response": response_text,
            "total_tokens": total_tokens
        }

        # Write the data to a JSON file
        with open(f"{self.log_dir}/{filename}", "w") as f:
            json.dump(data, f)

    def make_messages(self, prompt: str, system_instruction=None) -> List[Dict[str, str]]:
        """
        Create a list of messages in the format expected by the OpenAI Completion.create method.

        :param prompt: The user's input prompt.
        :return: A list of messages.
        """
        system_instruction = system_instruction or "You are an helpful assistant."
        return [{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]

    def _prompt_or_prompt_file(self, prompt=None, prompt_file=None, prompt_kwargs=None):

        if not prompt and not prompt_file:
            raise AttributeError('One of prompt and prompt_file must be defined.')

        if prompt and prompt_file:
            raise AttributeError('Only one of prompt and prompt_file can be defined.')

        if not prompt:
            with open(prompt_file, "r") as f:
                prompt = f.read()

        if prompt_kwargs:
            prompt = prompt.format(**prompt_kwargs)

        return prompt

    def _prompt_or_messages(self, prompt=None, messages=None, system_instruction=None):
        if not prompt and not messages:
            raise AttributeError('At least one of prompt and messages must be defined')

        if prompt and messages:
            messages.append({"role": "user", "content": prompt})
            messages[0]["content"] = system_instruction or messages[0]["content"]

        if not messages:
            messages = self.make_messages(prompt, system_instruction)

        return messages

    def make_payload(self, prompt=None, prompt_file=None, prompt_kwargs=None, messages=None, system_instruction=None, payload_kwargs=None):
        prompt = self._prompt_or_prompt_file(prompt, prompt_file, prompt_kwargs)
        messages = self._prompt_or_messages(prompt, messages, system_instruction)
        payload_kwargs = payload_kwargs or {}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        payload.update(self.completion_kwargs)
        payload.update(payload_kwargs)
        return payload

    @staticmethod
    async def send_request(payload, api_url, headers):
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(api_url, headers=headers, data=json.dumps(payload))

            if response.status_code == 200:
                return response.json()
            else:
                print(f"Request failed with status code {response.status_code}")
                return None

    async def send_requests(self, payloads, api_url, headers):
        tasks = [self.send_request(payload=payload, api_url=api_url, headers=headers) for payload in payloads]
        return await asyncio.gather(*tasks)

    def get_responses(
            self,
            payloads,
            api_url=None,
            api_key=None,
            eval_function=None,
            eval_function_kwargs=None,
            eval_iterables_dict=None,
            eval_valid_exceptions=None,
            max_eval_retrials=3
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY") or getattr(openai, "api_key", None)

        if not api_key:
            raise ValueError("API key not provided and not set in environment or openai library")

        api_url = api_url or "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        single_payload = not isinstance(payloads, list)
        if single_payload:
            payloads = [payloads]
        else:
            payloads, index_map = flatten_list(payloads)
            for payload in payloads:
                payload["stream"] = False

        responses = asyncio.run(self.send_requests(payloads, api_url, headers))
        response_texts = []
        for response, payload in zip(responses, payloads):
            total_tokens = response['usage']['total_tokens']
            response_text = response['choices'][0]['message']['content']
            messages = payload['messages']
            self.log_completions(messages, response_text, total_tokens)
            self.token_count += total_tokens
            response_texts.append(response_text)

        if eval_function:
            eval_function_kwargs = eval_function_kwargs or {}
            response_processor = ResponseProcessor(
                eval_function=eval_function,
                eval_function_kwargs=eval_function_kwargs,
                eval_iterables_dict=eval_iterables_dict,
                eval_valid_exceptions=eval_valid_exceptions,
            )
            for i, response_text in response_processor(list(range(len(response_texts))), response_texts, payloads):
                response_texts[i] = response_text

            if max_eval_retrials:
                _payloads, _idx_map, _eval_iterables_dict = response_processor.reset()
                if _payloads:
                    print(f"\n\nStarting Retrial ({max_eval_retrials-1} retrials left)")
                    _response_texts = self.get_responses(
                        _payloads,
                        api_url=api_url,
                        api_key=api_key,
                        eval_function=eval_function,
                        eval_function_kwargs=eval_function_kwargs,
                        eval_iterables_dict=_eval_iterables_dict,
                        eval_valid_exceptions=eval_valid_exceptions,
                        max_eval_retrials=max_eval_retrials-1
                    )
                    for original_idx, response_text in zip(_idx_map, _response_texts):
                        print('Level:', max_eval_retrials, 'Index:', original_idx)
                        response_texts[original_idx] = response_text

        if single_payload:
            return response_texts[0]
        else:
            return unflatten_list(response_texts, index_map)
