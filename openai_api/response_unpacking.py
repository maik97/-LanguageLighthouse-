

class UnpackingResponseError(Exception):
    pass


def unpack_completion(payload, response):
    if not response:
        raise UnpackingResponseError('Response was empty or None.')
    if 'choices' not in response:
        raise UnpackingResponseError('Response does not contain completion.')
    return (
        response['choices'][0]['message']['content'],
        payload['messages'],
        response['usage']['total_tokens']
    )