from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://10.1.47.245:8000/v1")

messages = [
    {"role": "system", "content": "You are a strict JSON-only extractor."},
    {"role": "user", "content": 'TASK: Diagnose and resolve the battery pack A2 over-temperature alarm.\n\nReturn exactly:\n{"core_incident_objects": ["..."]}'}
]

resp = client.chat.completions.create(
    model="/home/skyhe666/llm-model/Qwen/Qwen3.5-9B",
    messages=messages,
    temperature=0.0,
    max_tokens=2048,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

msg = resp.choices[0].message
print("finish_reason:", resp.choices[0].finish_reason)
print("Response content:", repr(msg.content))
