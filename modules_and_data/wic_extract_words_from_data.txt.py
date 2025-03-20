def extract_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = [line.split('\t')[0] for line in file if line.strip()]  # Az első oszlopot vesszük ki
    return words

# Define which dataset you want to work with
actual_working_dataset = 'dev'

# Fájl beolvasása és a szavak kiírása
file_path = f'{actual_working_dataset}.data.txt'
words = extract_words(file_path)
print('\n'.join(words))

with open(f'{actual_working_dataset}_extracted_words.txt', 'w', encoding='utf-8') as outfile:
    outfile.write('\n'.join(words))
