import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Is cuda available?", torch.cuda.is_available())
MODEL_NAME =  '       meta-llama/Llama-2-7b-chat-hf            '

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME.strip())
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def main():
    messages = [
        {"role": "system", "content": "Answer all questions with Yes or No!"},
        {"role": "user", "content": open("prompt.txt").read()},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)

if __name__ == '__main__':
    main()