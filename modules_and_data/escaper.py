import re
import os

# Define which dataset you want to work with
actual_working_dataset = 'test'

# Correct the file path assignment (removed trailing comma)
actual_file_name = f'formatted_{actual_working_dataset}_dataset.txt'

# Escape non-beginning and non-ending single quotes
def escape_inner_quotes(text):
    return re.sub(r"(?<!^)\'(?!$)", r"\\'", text)

# Check if the file is already escaped
def is_already_escaped(text):
    return r"\'" in text  # Checks for already escaped content

# Process the file
with open(actual_file_name, 'r', encoding='utf-8') as file:
    content = file.read()

if is_already_escaped(content):
    print(f"Skipping {actual_file_name}: Already escaped.")
else:
    escaped_content = escape_inner_quotes(content)
    with open(actual_file_name, 'w', encoding='utf-8') as file:
        file.write(escaped_content)
    print(f"Escaped quotes in: {actual_file_name}")
