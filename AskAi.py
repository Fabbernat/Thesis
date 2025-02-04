from WordSenseDisambiguator import sample_questions, WordSenseDisambiguator

# 1
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = model.to(device)

# 2
wsd_model = WordSenseDisambiguator()
input_text = sample_questions(wsd_model)
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