from datasets import load_dataset
from openai import OpenAI

api_key = "sk-hOp1BZAZ7edZTMQ390bO7jIiZghdoRmeJHOf4y84rbOey0Yn"
base_url = "https://yeysai.com/v1/messages"
model = "claude-opus-5"

client = OpenAI(api_key = api_key, base_url = base_url)
response = client.chat.completions.create(
    model = model,
    messages = [

    ]
)

print(response.choices[0].message.content) 