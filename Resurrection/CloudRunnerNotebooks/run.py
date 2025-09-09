import torch

print("Is cuda available?", torch.cuda.is_available())

MODEL_NAME =  '      https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct           '

from transformers import AutoTokenizer, AutoModelForCausalLM


def run():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME.strip())
    except Exception as exc:
        print("Exception while trying to load tokenizer:", exc)
    except Error as err:
        print("Error while trying to load tokenizer:", err)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    messages = [
        {"role": "system", "content": "Answer all questions with Yes or No!"},
        {"role": "user", "content": open("prompt.txt").read()},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    with open("modelAnswers.out", "w") as modelAnswersFile:
        print(answer, file=modelAnswersFile)
        # print((line for line in answer), file=modelAnswersFile)