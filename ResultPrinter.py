def print_results(synonyms, model, questions):
    """
        passes the database
    :param questions:
    :param model:
    :param synonyms:
    :return:
    """

    correct_answers_count = 0
    total_questions = 0
    results = {}

    for key, value in questions.items():
        model_answer = model.process_question(key, synonyms)
        correct_answers_count += (model_answer == value)
        answer = 'YES' if model_answer == value else 'NO'
        print(f'Sentence: "{key}"')
        print(f'Did the model predict correctly? {answer}')

    print(f'accuracy = {correct_answers_count / (len(questions) + 1e-5)}')
    print(f'{correct_answers_count} correct answer(s) out of {len(questions)} answers.')