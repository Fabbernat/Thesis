import re

# Define which dataset you want to work with
actual_working_dataset = 'dev'

# List of file paths to process
actual_file_name = f'formatted_{actual_working_dataset}_dataset.txt'


# Escape non-beginning and non-ending single quotes
def escape_inner_quotes(text):


# Process the file
with open(actual_file_name, 'r', encoding='utf-8') as file:
    content = file.read()

escaped_content = escape_inner_quotes(content)

with open(actual_file_name, 'w', encoding='utf-8') as file:
    file.write(escaped_content)

print(f"Escaped quotes in: {actual_file_name}")

