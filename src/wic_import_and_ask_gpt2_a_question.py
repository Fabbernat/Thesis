# C:\PycharmProjects\Peternity\src\wic_import_and_ask_gpt2_a_question.py
import llm_prompts.scripts.obvious_questions

# 1
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)

# 2
input_text = sample_questions()
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

print(input_ids, "\n")

for id in input_ids[0]:
  print(id, tokenizer.decode(id, skip_special_tokens=True))

# 3
output = model.generate(input_ids,
                        max_length=50,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id)


generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)