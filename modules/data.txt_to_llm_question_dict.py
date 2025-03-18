import wic_sentence_normalizer
import re

# Read the .txt files
with open('test.data.txt', 'r', encoding='utf-8') as data_file, \
        open('test.gold.txt', 'r', encoding='utf-8') as gold_file:
    data_lines = data_file.readlines()
    gold_lines = gold_file.read().strip().split('\n')  # Read and split labels

# Ensure data and labels have the same length
if len(data_lines) != len(gold_lines):
    raise ValueError("Mismatch between data lines and gold labels")

# Process the data
data = []
for line, label in zip(data_lines, gold_lines):
    parts = line.strip().split('\t')
    if len(parts) == 5:  # Ensure data integrity
        word, pos, freq, sentence1, sentence2 = parts
        sentence1 = wic_sentence_normalizer.make_sentence_human_readable(sentence1)
        sentence2 = wic_sentence_normalizer.make_sentence_human_readable(sentence2)
        answer = 'Yes' if label.strip() == 'T' else 'No'
        formatted = f"'Does the word \"{word}\" mean the same thing in sentences \"{sentence1}\" and \"{sentence2}\"?': '{answer}',"
        data.append(formatted)

# Write the output to a new file
with open('formatted_data.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(data))

print("Data formatting complete. Check 'formatted_data.txt'.")
