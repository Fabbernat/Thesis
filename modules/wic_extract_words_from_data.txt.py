def extract_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = [line.split('\t')[0] for line in file if line.strip()]  # Az első oszlopot vesszük ki
    return words

# Fájl beolvasása és a szavak kiírása
file_path = 'test.data.txt'
words = extract_words(file_path)
print('\n'.join(words))
