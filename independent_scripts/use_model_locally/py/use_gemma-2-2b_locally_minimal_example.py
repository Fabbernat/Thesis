from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

input_text = "Does 'defeat' mean the same in these sentences? Answer Yes or No."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))