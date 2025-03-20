# Word Sense Disambiguation module
# https://pilehvar.github.io/wic/
import os
import re
from typing import Set, Optional, Dict, List

import wic_result_printer
from modules import wic_nltk_handler


def build_sentence_simple(word: str, sentence_a: str, sentence_b: str) -> str:
    """Builds a structured question from word and sentences."""
    return f'Does the word "{word}" mean the same thing in sentences "{sentence_a}" and "{sentence_b}"?'


def build_sentence(word: str, pos: str, index1, index2, sentence_a: str, sentence_b: str) -> str | bool:
    try:
        index1 = int(index1)
        index2 = int(index2)
    except Exception | TypeError:
        return False
    if pos == 'V':
        part_of_speech = 'verb'
    else:
        part_of_speech = 'noun'
    """Builds a structured question from word and sentences."""
    return f'Does the word "{word}" mean the same thing in sentence "{sentence_a}" -  at position {index1} - and in sentence"{sentence_b}" - at position {index2} - ? The word "{word}" is a {part_of_speech} in both sentences.'


def get_word_context(word: str, sentence: str, synonyms: Optional[Set[str]] = None) -> Set[str]:
    """Returns surrounding words of the target word in a sentence, optionally including synonyms."""
    words = sentence.split()
    context = set()

    if word in words:
        index = words.index(word)
        left_context = words[max(0, index - 5): index]  # Get up to 5 words before
        right_context = words[index + 1: index + 5]  # Get up to 5 words after
        context.update(left_context + right_context)

    if synonyms:
        context.update(synonyms)

    return context


def determine_word_similarity(word: str, sentence_a: str, sentence_b: str,
                              synonyms: Optional[Set[str]]) -> str:
    """Determines if the word has the same meaning in two different sentences based on context similarity."""
    context_a = get_word_context(word, sentence_a, synonyms)
    context_b = get_word_context(word, sentence_b, synonyms)

    similarity = len(context_a & context_b) / (len(context_a | context_b) + 1e-5)  # Avoid division by zero

    return "YES" if similarity > 0 else "NO"


def process_question(question: str, synonyms: Optional[Set[str]]) -> str:
    """Extracts components from the formatted question and determines the answer."""
    pattern = r'Does the word "(.+?)" mean the same thing in sentences "(.+?)" and "(.+?)"\?'
    match = re.match(pattern, question)

    if not match:
        return "Invalid question format"

    word, sentence_a, sentence_b = match.groups()
    return determine_word_similarity(word, sentence_a, sentence_b, synonyms)



# Example usage:
'''
word = "run"
sentence_a = "I went for a run in the park."
sentence_b = "The play had a long run on Broadway."
context_a = {"for", "a", "in", "the"}
context_b = {"a", "long", "on", "Broadway"}
similarity = len(context_a & context_b) / (len(context_a | context_b) + 1e-5)
common_words = context_a & context_b = {"a"}
total_words = context_a | context_b = {"for", "a", "in", "the", "long", "on", "Broadway"}

similarity = len({"a"}) / len({"for", "a", "in", "the", "long", "on", "Broadway"})
           = 1 / 7
           = 0.14
return "YES" if similarity > 0 else "NO"
'''


def read_wic_dataset(base_path: str, file_specific_path):
    """
    Reads the WiC dataset from the given base directory and returns a structured dictionary.

    :param base_path: The root directory containing the WiC dataset.
    :? : ?
    :return: A dictionary with 'train', 'dev', and 'test' datasets.
    """
    full_path = os.path.join(base_path, file_specific_path)

    entries = []

    if not os.path.exists(full_path):
        print(f"Skipping: {full_path} (file not found)")
        return entries

    with open(full_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                print(f"Skipping invalid line in {file_specific_path}: {line}")
                continue

            word, pos, index, sentence_a, sentence_b = parts[:5]
            entries.append({
                "word": word,
                "pos": pos,
                "index": index,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b
            })
    return entries



def process_wic_data(wic_data: Dict[str, List[Dict[str, str]]]) -> dict[str, dict[str, str]]:
    """
    Processes the WIC data read from
    and builds sentences for evaluation.
    :wic_data: contains train, dev and test sets
    :rtype: object
    """
    questions = {}  # Dictionary to hold data for each split
    if wic_data:
        i = 0
        for split in wic_data:  # Iterate through 'train', 'dev', 'test' splits
            split_data = {}  # Dict to hold rows of data
            i += 1
            for row in wic_data[split]:
                word = row['word']
                sentence_a = row['sentence_a']
                sentence_b = row['sentence_b']
                label = row['label']

                split_data['word'] = word
                split_data['sentence_a'] = sentence_a
                split_data['sentence_b'] = sentence_b
                split_data['label'] = label

            print('BEGIN\n\n', split, '\n\nEND')
            questions[split] = split_data
    return questions


def sample_questions():
    """

    :param model:
    :return: questions
    """
    # hardcoded add
    questions: Dict[str, str] = {
        'Does the word "run" mean the same thing in sentences "I went for a run in the park." and "The play had a long run on Broadway."?': 'NO',
        'Does the word "defeat" mean the same thing in sentences "It was a narrow defeat." and "The army\'s only defeat."?': 'NO',
        'Does the word "bank" mean the same thing in sentences "Bank on your good education." and "The pilot had to bank the aircraft."?': 'YES'
    }

    # modular add
    # 1
    word = 'penetration'

    sentence_a = 'The penetration of upper management by women .'
    sentence_b = 'Any penetration , however slight , is sufficient to complete the offense .'

    built_sentence = build_sentence_simple(word, sentence_a, sentence_b)

    questions[built_sentence] = 'NO'

    # 2
    word = 'penetrate'
    sentence_a = 'The hikers did not manage to penetrate the dense forest .'
    sentence_b = 'She was penetrated with sorrow .'

    built_sentence = build_sentence_simple(word, sentence_a, sentence_b)

    questions[built_sentence] = 'YES'

    return questions





def read_gold_labels(base_path: str, file_specific_path: str) -> List[str]:
    """
    Reads a gold label file and returns a list of labels.
    """
    full_path = os.path.join(base_path, file_specific_path)
    labels = []

    if not os.path.exists(full_path):
        print(f"Skipping: {full_path} (file not found)")
        return labels

    with open(full_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f]
    return labels


def load_wic_data(base_dir: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Loads all WiC dataset files into a structured dictionary.

    :param base_dir: The root directory containing the WiC dataset.
    :return: A dictionary with 'train', 'dev', and 'test' datasets.
    """
    dataset_splits = ["train", "dev", "test"]
    wic_data: Dict[str, List[Dict[str, str]]] = {}

    for split in dataset_splits:
        data = read_wic_dataset(base_dir, f"{split}/{split}.data.txt")
        gold_labels = read_gold_labels(base_dir, f"{split}/{split}.gold.txt")

        if len(data) == len(gold_labels):
            for i, entry in enumerate(data):
                entry["label"] = gold_labels[i]
        else:
            print(f"Warning: Mismatched data and label count for {split}")

        wic_data[split] = data

    return wic_data

def main():
    # Run the model with no synonyms
    print()
    print('"dumb" algorithm implemented by Fabbernat:')
    wic_result_printer.print_results(None, sample_questions())

    print()
    print('nltk wordnet algorithm:')
    wic_result_printer.print_results(wic_nltk_handler.synonyms, sample_questions())

    base_dir = r'C:\WiC_dataset'

    wic_data = load_wic_data(base_dir)
    questions = {}

    processed_data = process_wic_data(wic_data)
    print(processed_data)
    for row in processed_data.values():
        print(row)
        # for entry in row.values():
        #     entry : Tuple[str, str, str, str] = entry
        #     assert isinstance(entry, tuple) and len(entry) == 4
        word = row.get('word')
        sentence_a = row.get('sentence_a')
        sentence_b = row.get('sentence_b')
        label = row.get('label')
        built_sentence = build_sentence_simple(word, sentence_a, sentence_b)
        questions[built_sentence] = 'YES' if label == 'T' else 'NO'

        wic_result_printer.print_results(wic_nltk_handler.synonyms, questions)


if __name__ == '__main__':
    main()