# I made some tricky questions that are not part of any of the dev/train/test datasets

unchecked_questions: Dict[str, str] = {
    'Does the word "bank" mean the same in sentence "The treasure sunk onto the river bank." and "The treasure chamber of the financial building is beneath the river bank."?': 'No',
    'Does the word "running" mean the same in sentence "My program is running." and "My nose is running."?': 'No',
    'Does the word "running" mean the same in sentence "My program is running." and "My brother is running."?': 'No',
    'Does the word "running" mean the same in sentence "My nose is running." and "My brother is running."?': 'No',
    'Does the word "approach" mean the same thing in sentences "Would counsel please approach the bench? asked the judge." and "He approached the age of manhood."?': 'No',
    'Does the word "approval" mean the same thing in sentences "He bought it on approval." and "Although she fussed at them, she secretly viewed all her children with approval."?': 'No',
    'Does the word "bandage" mean the same thing in sentences "Bandage an incision." and "The nurse bandaged a sprained ankle."?': 'Yes',
    'Does the word "beat" mean the same thing in sentences "He heard the beat of a drum." and "The conductor set the beat."?': 'Yes',
    'Does the word "bell" mean the same thing in sentences "Saved by the bell." and "She heard the distant toll of church bells."?': 'No',
    'Does the word "bit" mean the same thing in sentences "A bit of paper." and "A bit of rock caught him in the eye."?': 'Yes',
    'Does the word "chip" mean the same thing in sentences "Be careful not to chip the paint." and "Chip a tooth."?': 'Yes',
    'Does the word "conduct" mean the same thing in sentences "You can\'t conduct business like this." and "To conduct the affairs of a kingdom."?': 'Yes',
    'Does the word "cure" mean the same thing in sentences "Cure meats." and "Cure hay."?': 'Yes',
    'Does the word "talking" mean the same thing in sentences "We spent hours talking about our future." and "The parrot was talking in a human-like voice."?': 'Yes',
    'Does the word "school" mean the same thing in sentences "She goes to school every weekday." and "A school of fish swam past the boat."?': 'No',
    'Does the word "lawyer" mean the same thing in sentences "She hired a lawyer to represent her in court." and "He was his own lawyer in the trial."?': 'Yes',
}
