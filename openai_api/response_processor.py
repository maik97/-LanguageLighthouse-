from collections import defaultdict


def iterable_dict_generator(iterables_dict):
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
        if eval_iterables_dict:
            self.iterable_eval_kwargs = [subdict for subdict in iterable_dict_generator(eval_iterables_dict)]
        else:
            self.iterable_eval_kwargs = None
        self.valid_exceptions = eval_valid_exceptions or (Exception,)

    def __call__(self, idx_map, responses, payloads):
        for idx, response, payload in zip(idx_map, responses, payloads):
            try:
                eval_kwargs = {}
                if self.eval_function_kwargs:
                    eval_kwargs.update(self.eval_function_kwargs.copy())
                if self.iterable_eval_kwargs:
                    eval_kwargs.update(self.iterable_eval_kwargs[idx].copy())
                response = self.eval_function(response, **eval_kwargs)
                yield idx, response
            except self.valid_exceptions as e:
                print('\033[35m\nResponse Evaluation Failed.\033[0m')
                print('\033[35mException:\n\033[0m', e)
                print('\033[35mIndex:\n\033[0m', idx)
                print('\033[35meval_kwargs:\n\033[0m', eval_kwargs)
                print('\033[35mResponse:\n\033[0m', response)
                print('\033[35mPayload:\n\033[0m', payload)

                self.failed_payloads.append(payload)
                self.failed_payloads_index.append(idx)

    def reset(self):
        failed_payloads, failed_payloads_index = self.failed_payloads, self.failed_payloads_index
        self.failed_payloads = []
        self.failed_payloads_index = []

        if self.iterable_eval_kwargs:
            eval_iterables_dict = defaultdict(list)
            for idx in failed_payloads_index:
                for key, value in self.iterable_eval_kwargs[idx].items():
                    eval_iterables_dict[key].append(value)
        else:
            eval_iterables_dict = None

        return failed_payloads, failed_payloads_index, eval_iterables_dict
