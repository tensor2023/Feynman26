from feynman_adapter import api_call

text, usage = api_call([{"role": "user", "content": "Reply with exactly OK"}], max_tokens=8, thinking=False)
print(text.strip())
print({key: usage.get(key) for key in ("prompt_tokens", "completion_tokens")})
