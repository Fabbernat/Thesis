from WordSenseDisambiguator import process_question


def print_results(synonyms, questions):
    """
        passes the database
    :param questions:
    :param synonyms:
    :return:
    """

    correct_answers_count = 0
    results = {}

    for key, value in questions.items():
        model_answer = process_question(key, synonyms)
        correct_answers_count += (model_answer == value)
        answer = 'YES' if model_answer == value else 'NO'
        print(f'Sentence: "{key}"')
        print(f'Did the model predict correctly? {answer}')

    if len(questions) > 0:
        print(f'accuracy = {correct_answers_count / len(questions)}')
    print(f'{correct_answers_count} correct answer(s) out of {len(questions)} answers.')