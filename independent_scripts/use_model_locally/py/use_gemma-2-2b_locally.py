import torch
from transformers import pipeline

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Adjust the path to match your operating system!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Windows
model_path = r'C:\codes\gemma22b\gemma-2-2b-it'

try:
    pipe = pipeline(
        'text-generation',
        model=model_path,
        model_kwargs={'torch_dtype':torch.bfloat16},
        device='cuda', # replace with 'mps' on a Mac device
    )

    question = '''
    Answer the question with Yes or No with reasoning!
    Does the word "defeat" mean the same thing in sentences "It was a narrow defeat." and "The army's only defeat."?
    '''

    messages = [{'role':'user', 'content': question}]

    outputs = pipe(messages, max_new_tokens=4096)
    assistant_response = outputs[0]['generated_text'][-1]['content'].strip()
    print(assistant_response)
except Exception as e:
    print(f"Error: {e}")
    # Fallback to CPU if CUDA fails
    pipe = pipeline('text-generation', model=model_path, device='cpu')
    output = pipe(question, max_new_tokens=50)
    print(output[0]['generated_text'])