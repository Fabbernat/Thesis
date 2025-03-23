# Define which dataset you want to work with
actual_working_dataset = 'test'

# Read the .txt files
with open(f'{actual_working_dataset}.data.txt', 'r', encoding='utf-8') as data_file, \
        open(f'{actual_working_dataset}.gold.txt', 'r', encoding='utf-8') as gold_file:
    data_lines = data_file.readlines()
    gold_lines = gold_file.read().strip().split('\n')  # Read and split labels

# Ensure data and labels have the same length
if len(data_lines) != len(gold_lines):
    raise ValueError("Mismatch between data lines and gold labels")


def make_sentence_human_readable(sentence):
    """Replaces contractions for better readability both for humans and for language models."""
    sentence = sentence.replace(" 's", "\'s")
    sentence = sentence.replace("'", "\\'")  # Escape all single quotes
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace(" .", ".")
    return sentence


# Process the data
data = []
for line, label in zip(data_lines, gold_lines):
    parts = line.strip().split('\t')
    if len(parts) == 5:  # Ensure data integrity
        word, pos, freq, sentence1, sentence2 = parts
        sentence1 = make_sentence_human_readable(sentence1)
        sentence2 = make_sentence_human_readable(sentence2)
        answer = 'Yes' if label.strip() == 'T' else 'No'

        # Reverse the order by swapping sentence2 and sentence1
        formatted = f"r'Does the word \"{word}\" mean the same thing in sentences \"{sentence2}\" and \"{sentence1}\"?': '{answer}',"
        data.append(formatted)

# Write the output to a new file
with open(f'formatted_{actual_working_dataset}_dataset_reversed.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(data))

print("Data formatting complete. Check 'formatted_data.txt'.")


# Example
print(make_sentence_human_readable(
    "We had to swim for 20 minutes to reach the shore . A big fish was swimming in the tank . Do n't fire until you see the whites of their eyes . The gun fired ."))
