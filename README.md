# LanguageLighthouse
Language Lighthouse is a library for the OpenAI chat completion API using state-of-the-art language models like GPT, with support for asynchronous calls.

# Example
``` python
  from async_gpt import AsyncGPT
  
  gpt = AsyncGPT()
  
  payloads = []
  for text in texts:
      payloads.append(gpt.make_payload(
          prompt="Either this or prompt file."
          prompt_file=os.path.join(PROMPTS_DIR, 'prompt.txt'),
          prompt_kwargs=dict(
              input_text=text,
          )
      ))

  responses= gpt.get_responses(
      payloads=payloads,
  )
```
