import anthropic

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="my_api_key",
)

message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=20000,
    temperature=1,
    system="Answer questions with a single Yes or No.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Does the word \"dog\" mean the same thing in sentences \"The dog barked.\" and \"The dog wagged its tail.\"?\nDoes the word \"apple\" mean the same thing in sentences \"I ate an apple.\" and \"He owns Apple Inc.\"?"
                }
            ]
        }
    ]
)
print(message.content)