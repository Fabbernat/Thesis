# C:\PycharmProjects\Peternity\modules_and_data\modules\wic_extract_sentences_from_data.txt.py
def extract_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [(line.split('\t')[3], line.split('\t')[4]) for line in file if line.strip()]  # Az első oszlopot vesszük ki
    return sentences

# Define which dataset you want to work with
actual_working_dataset = 'test'

# Fájl beolvasása és a szavak kiírása
file_path = f'../data/txt/{actual_working_dataset}.data.txt'
sentences = extract_sentences(file_path)

output_text = '\n\n'.join(f'{s1}\n{s2}' for s1, s2 in sentences)

print(output_text)

with open(f'../data/txt/{actual_working_dataset}_extracted_sentences.txt', 'w', encoding='utf-8') as outfile:
    outfile.write(output_text)
