# C:\PycharmProjects\Peternity\src\wic_result_printer.py
from src.utils.wic_sentence_normalizer import make_sentence_human_readable


def print_results(synonyms, questions):
    correct_answers_count = 0

    for key, value in questions.items():
        from src.utils.wic_word_sense_disambiguator import process_question
        model_answer = process_question(key, synonyms)
        correct_answers_count += (model_answer == value)
        answer = 'YES' if model_answer == value else 'NO'
        key = make_sentence_human_readable(key)
        print(f'Sentence: "{key}"')
        print('model answer:', model_answer)
        print('actual answer:', value)
        print(f'Did the model predict correctly? {answer}')

    if len(questions) > 0:
        print(f'accuracy = {correct_answers_count / len(questions)}')
    print(f'{correct_answers_count} correct answer(s) out of {len(questions)} answers.')