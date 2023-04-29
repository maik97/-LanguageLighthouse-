import asyncio
import json
import os
from datetime import datetime
import time

import httpx
import openai
from typing import List, Dict, Tuple, Optional

from jinja2 import Template

from openai_api.response_processor import ResponseProcessor
from openai_api.response_unpacking import unpack_completion
from utilities import nestlings
from utilities.lazy_init import LazyInit


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
        self.wait = 0

    def prompt_test(
            self,
            prompt=None,
            prompt_file=None,
            prompt_template=None,
            prompt_kwargs=None,
            messages=None,
            system_instruction=None,
            payload_kwargs=None,
            debug_prints=False
    ):
        payload = self.make_payload(
            prompt=prompt,
            prompt_file=prompt_file,
            prompt_template=prompt_template,
            prompt_kwargs=prompt_kwargs,
            messages=messages,
            system_instruction=system_instruction,
            payload_kwargs=payload_kwargs
        )

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

    def _prompt_or_prompt_file(self, prompt=None, prompt_file=None, prompt_template=None, prompt_kwargs=None):

        # use text file:
        if prompt_file and not prompt and not prompt_template:
            with open(prompt_file, "r") as f:
                prompt = f.read()

        # use ninja2 template:
        elif prompt_template and not prompt and not prompt_file:
            with open(prompt_template, "r") as f:
                prompt = f.read()
            prompt = Template(prompt)

        # use prompt, check it is defined as the only attribute
        elif prompt and not (prompt or prompt_file):
            pass
        else:
            AttributeError('Exactly one of prompt, prompt_file or prompt_template must be defined.')

        prompt_kwargs = prompt_kwargs or {}
        if isinstance(prompt, str) and prompt_kwargs:
            prompt = prompt.format(**prompt_kwargs)
        elif prompt_kwargs:
            prompt = prompt.render(**prompt_kwargs)

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

    def make_payload(
            self,
            prompt=None,
            prompt_file=None,
            prompt_template=None,
            prompt_kwargs=None,
            messages=None,
            system_instruction=None,
            payload_kwargs=None,
    ):
        prompt = self._prompt_or_prompt_file(prompt, prompt_file, prompt_template, prompt_kwargs)
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

    async def send_request(self, payload, api_url, headers, timeout=300.0):
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, headers=headers, data=json.dumps(payload))

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                self.wait = time.time() + 60
            else:
                print(f"Request failed with status code {response.status_code}")
                return None

    async def send_requests(self, payloads, api_url, headers, timeout):
        request_kwargs = dict(
            api_url=api_url,
            headers=headers,
            timeout=timeout
        )
        tasks = [self.send_request(payload=payload, **request_kwargs) for payload in payloads]
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
            max_eval_retrials=3,
            timeout=300.0
    ):

        # Prepare Requests:
        if not payloads:
            return payloads

        while self.wait > time.time():
            print('Waiting:', self.wait, time.time())
            time.sleep(1)

        api_key = api_key or os.getenv("OPENAI_API_KEY") or getattr(openai, "api_key", None)

        if not api_key:
            raise ValueError("API key not provided and not set in environment or openai_api library")

        api_url = api_url or "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Send Requests:
        payloads, blueprint = nestlings.flatten(payloads, make_blueprint=True, exclude=(dict,))
        for payload in payloads:
            payload["stream"] = False

        responses = asyncio.run(self.send_requests(payloads, api_url, headers, timeout=timeout))
        response_texts = []
        for response, payload in zip(responses, payloads):
            try:
                response_text, messages, total_tokens = unpack_completion(payload, response)
                self.log_completions(messages, response_text, total_tokens)
                self.token_count += total_tokens
                response_texts.append(response_text)
            except UnpackingResponseError:
                response_texts.append(None)

        # Evaluate Requests:
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
                    print(f"\n\nStarting Retrial ({max_eval_retrials - 1} retrials left)", self.token_count)
                    _response_texts = self.get_responses(
                        _payloads,
                        api_url=api_url,
                        api_key=api_key,
                        eval_function=eval_function,
                        eval_function_kwargs=eval_function_kwargs,
                        eval_iterables_dict=_eval_iterables_dict,
                        eval_valid_exceptions=eval_valid_exceptions,
                        max_eval_retrials=max_eval_retrials - 1,
                        timeout=timeout,
                    )
                    for original_idx, response_text in zip(_idx_map, _response_texts):
                        print('Level:', max_eval_retrials, 'Index:', original_idx)
                        response_texts[original_idx] = response_text

        return nestlings.construct(blueprint, response_texts)


class LazyAsyncChatGPT:
    gpt = LazyInit(AsyncChatGPT)

