from typing import Dict

human_readable_questions_short: Dict[str, str] = {

}

unchecked_questions = {

}

human_readable_questions_long = {}
human_readable_questions_full = {}

selected_questions = human_readable_questions_short
with_reasoning = ""
explain = True
if explain:
    with_reasoning = ' with reasoning'

print(f'Answer all {len(selected_questions)} questions with Yes or No{with_reasoning}!')
print(*human_readable_questions_short.keys(), sep='\n')
