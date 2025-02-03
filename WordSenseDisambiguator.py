# Word Sense Disambiguation module

import re
import mynltk
import os
from typing import Set, Optional, Dict

class WordSenseDisambiguator:

    def get_word_context(self, word: str, sentence: str, synonyms: Optional[Set[str]] = None) -> Set[str]:
        """Returns surrounding words of the target word in a sentence, optionally including synonyms."""
        words = sentence.split()
        context = set()

        if word in words:
            index = words.index(word)
            left_context = words[max(0, index - 2): index]  # Get up to 2 words before
            right_context = words[index + 1: index + 3]  # Get up to 2 words after
            context.update(left_context + right_context)

        if synonyms:
            context.update(synonyms)

        return context

    def determine_word_similarity(self, word: str, sentence_a: str, sentence_b: str,
                                  synonyms: Optional[Set[str]]) -> str:
        """Determines if the word has the same meaning in two different sentences based on context similarity."""
        context_a = self.get_word_context(word, sentence_a, synonyms)
        context_b = self.get_word_context(word, sentence_b, synonyms)

        similarity = len(context_a & context_b) / (len(context_a | context_b) + 1e-5)  # Avoid division by zero

        return "YES" if similarity > 0 else "NO"

    def process_question(self, question: str, synonyms: Optional[Set[str]]) -> str:
        """Extracts components from the formatted question and determines the answer."""
        pattern = r'Does the word "(.+?)" mean the same thing in sentences "(.+?)" and "(.+?)"\?'
        match = re.match(pattern, question)

        if not match:
            return "Invalid question format"

        word, sentence_a, sentence_b = match.groups()
        return self.determine_word_similarity(word, sentence_a, sentence_b, synonyms)

    def build_sentence(self, word: str, sentence_a: str, sentence_b: str) -> str:
        """Builds a structured question from word and sentences."""
        return f'Does the word "{word}" mean the same thing in sentences "{sentence_a}" and "{sentence_b}"?'


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

def read_WiC_dataset(base_path: str):
    """
        Reads the WiC dataset from the given base directory and returns a structured dictionary.

        :param base_path: The root directory containing the WiC dataset.
        :return: A dictionary with 'train', 'dev', and 'test' datasets.
    """
    datasets = ["train", "dev", "test"]
    data_structure = {}

    for dataset in datasets:
        data_file = os.path.join(base_path, dataset, f"{dataset}.data.txt")
        gold_file = os.path.join(base_path, dataset, f"{dataset}.gold.txt")

        if not os.path.exists(data_file) or not os.path.exists(gold_file):
            print(f"Skipping {dataset}: Missing files")
            continue

        entries = []
        with open(data_file, "r", encoding="utf-8") as df, open(gold_file, "r", encoding="utf-8") as gf:
            for line, label in zip(df, gf):
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    print(f"Skipping invalid line in {data_file}: {line}")
                    continue

                word, sentence1, sentence2 = parts[:3]
                label = label.strip()
                entries.append({
                    "word": word,
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "label": label
                })

        data_structure[dataset] = entries

    return data_structure

def use_model(synonyms):
    """
        passes the database
    :param synonyms:
    :return:
    """
    model = WordSenseDisambiguator()
    correct_answers_count = 0
    total_questions = 0
    results = {}

# hardcoded add
    questions: Dict[str, str] = {
        'Does the word "run" mean the same thing in sentences "I went for a run in the park." and "The play had a long run on Broadway."?': 'NO',
        'Does the word "defeat" mean the same thing in sentences "It was a narrow defeat." and "The army\'s only defeat."?': 'NO',
        'Does the word "bank" mean the same thing in sentences "Bank on your good education." and "The pilot had to bank the aircraft."?': 'YES'
    }

# modular add
    word = 'penetration'
    sentence_a = 'The penetration of upper management by women .'
    sentence_b = 'Any penetration , however slight , is sufficient to complete the offense .'

    built_sentence = model.build_sentence(word, sentence_a, sentence_b)
    questions[built_sentence] = 'NO'

    word = 'penetrate'
    sentence_a = 'The hikers did not manage to penetrate the dense forest .'
    sentence_b = 'She was penetrated with sorrow .'

    built_sentence = model.build_sentence(word, sentence_a, sentence_b)
    questions[built_sentence] = 'YES'
    for key, value in questions.items():
        model_answer = model.process_question(key, synonyms)
        correct_answers_count += (model_answer == value)
        answer = 'YES' if model_answer == value else 'NO'
        print(f'Sentence: "{key}"')
        print(f'Did the model predict correctly? {answer}')

    print(f'accuracy = {correct_answers_count / len(questions)}')


def main():
    # Run the model with no synonyms
    print('"dumb" algorithm implemented by Fabbernat:')
    use_model(None)
    print('\nnltk wordnet algorithm:')
    use_model(mynltk.synonyms)

    base_dir = r"C:\Users\Bernát\Downloads\WiC_dataset"
    wic_data = read_WiC_dataset(base_dir)

    # Print a sample
    print(wic_data["train"][:2])

if __name__ == '__main__':
    main()