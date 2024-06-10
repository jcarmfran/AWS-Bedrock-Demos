import boto3
import json


def format_llama_messages(messages):
    """Format messages for Llama-2 chat models.

    The model only supports 'system', 'user', and 'assistant' roles, starting with 'system', then 'user', and
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    prompt = []

    if messages[0]["role"] == "system":
        content = "".join(["<<SYS>>\n", messages[0]["content"], "\n<</SYS>>\n\n", messages[1]["content"]])
        messages = [{"role": messages[1]["role"], "content": content}] + messages[2:]

    for user, answer in zip(messages[::2], messages[1::2]):
        prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ", (answer["content"]).strip(), "</s>"])

    prompt.extend(["<s>", "[INST] ", (messages[-1]["content"]).strip(), " [/INST] "])

    return "".join(prompt)

user_prompt = "Write a poem on machine learning?"
system_prompt = "You a Shakespearian poet"
all_messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
formatted_prompt = format_llama_messages(all_messages)

max_tokens_to_generate = 256
payload = {"prompt": formatted_prompt, 
           "max_gen_len": max_tokens_to_generate, 
           "temperature": 0.1, 
           "top_p": 0.9
           }

modelId = 'meta.llama2-70b-chat-v1'
accept = 'application/json'
contentType = 'application/json'

bedrock = boto3.client(service_name="bedrock-runtime")

response = bedrock.invoke_model(body=json.dumps(payload),
                                modelId=modelId,
                                accept=accept,
                                contentType=contentType
                                )

response_body=json.loads(response.get("body").read())
response_text=response_body['generation']
print(response_text)