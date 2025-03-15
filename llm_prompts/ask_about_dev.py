from typing import Dict, Any

human_readable_questions_short: Dict[str, str] = {
    'Does the word "board" mean the same thing in sentences "Room and board." and "He nailed boards across the windows."?': 'No',
    'Does the word "circulate" mean the same thing in sentences "Circulate a rumor." and "This letter is being circulated among the faculty."?': 'No',
    'Does the word "hook" mean the same thing in sentences "Hook a fish." and "He hooked a snake accidentally."?': 'Yes',
    'Does the word "recreation" mean the same thing in sentences "For recreation he wrote poetry." and "Drug abuse is often regarded as a form of recreation."?': 'Yes',
    'Does the word "domesticity" mean the same thing in sentences "Making a hobby of domesticity." and "A royal family living in unpretentious domesticity."?': 'No',
    'Does the word "acquisition" mean the same thing in sentences "The child\'s acquisition of language." and "That graphite tennis racquet is quite an acquisition."?': 'No',
    'Does the word "meeting" mean the same thing in sentences "There was no meeting of minds." and "The meeting elected a chairperson."?': 'No',
    'Does the word "nude" mean the same thing in sentences "They swam in the nude." and "The marketing rule "nude sells" spread from verbal to visual mainstream media."?': 'No',
    'Does the word "mark" mean the same thing in sentences "He left an indelible mark on the American theater." and "It was in London that he made his mark."?': 'Yes',
    'Does the word "association" mean the same thing in sentences "Conditioning is a form of learning by association." and "Many close associations with England."?': 'No',
    'Does the word "inclination" mean the same thing in sentences "The alkaline inclination of the local waters." and "An inclination of his head indicated his agreement."?': 'No',
    'Does the word "glaze" mean the same thing in sentences "Glaze the bread with eggwhite." and "The potter glazed the dishes."?': 'Yes',
    'Does the word "piggyback" mean the same thing in sentences "An amendment to piggyback the current law." and "He piggybacked her child so she could see the show."?': 'No',
    'Does the word "pick" mean the same thing in sentences "To pick rags." and "Don\'t always pick on your little brother."?': 'No',
    'Does the word "belabor" mean the same thing in sentences "Belabor the obvious." and "She was belabored by her fellow students."?': 'No',
    'Does the word "tear" mean the same thing in sentences "He took the manuscript in both hands and gave it a mighty tear." and "There were big tears rolling down Lisa\'s cheeks."?': 'Yes',
    'Does the word "kill" mean the same thing in sentences "Kill the engine." and "He killed the ball."?': 'No',
    'Does the word "lecture" mean the same thing in sentences "Did you ever lecture at Harvard?" and "She lectured to the class about her travels."?': 'No',
    'Does the word "forge" mean the same thing in sentences "We decided to forge ahead with our plans." and "He forged ahead."?': 'Yes',
    'Does the word "assurance" mean the same thing in sentences "An assurance of help when needed." and "His assurance in his superiority did not make him popular."?': 'Yes',
    'Does the word "branch" mean the same thing in sentences "A branch of Congress." and "We have branches in all major suburbs."?': 'Yes',
    'Does the word "risk" mean the same thing in sentences "I cannot risk smoking." and "Why risk your life?"?': 'Yes',
    'Does the word "chemistry" mean the same thing in sentences "The chemistry of indigo." and "The chemistry of iron."?': 'Yes',

    'Does the word "char" mean the same thing in sentences "Among other native delicacies, they give you fresh char." and "I had to scrub the kitchen today, because the char couldn\'t come."?': 'Unknown',
    'Does the word "response" mean the same thing in sentences "This situation developed in response to events in Africa." and "His responses have slowed with age."?': 'Unknown',
    'Does the word "excuse" mean the same thing in sentences "That thing is a poor excuse for a gingerbread man. Hasn\'t anyone taught you how to bake?" and "He\'s a sorry excuse of a doctor."?': 'Unknown',
    'Does the word "bondage" mean the same thing in sentences "He sought release from his bondage to Satan." and "A self freed from the bondage of time."?': 'Unknown',
    'Does the word "catch" mean the same thing in sentences "The catch was only 10 fish." and "He shared his catch with the others."?': 'Unknown',
    'Does the word "shadiness" mean the same thing in sentences "There\'s too much shadiness to take good photographs." and "The shadiness of their transactions."?': 'Unknown',
    'Does the word "passage" mean the same thing in sentences "The outward passage took 10 days." and "She struggled to play the difficult passages."?': 'Unknown',
    'Does the word "daughter" mean the same thing in sentences "I already have a son, so I would like to have a daughter." and "Her daughter cared for her in her old age."?': 'Unknown',
    'Does the word "hold" mean the same thing in sentences "Please hold a table at Maxim\'s." and "Hold a table for us at 7:00."?': 'Unknown',
    'Does the word "banish" mean the same thing in sentences "Banish bad thoughts." and "Banish gloom."?': 'Unknown',
    'Does the word "sense" mean the same thing in sentences "A keen musical sense." and "A good sense of timing."?': 'Unknown',
    'Does the word "opinion" mean the same thing in sentences "In my opinion, white chocolate is better than milk chocolate." and "I would like to know your opinions on the new systems."?': 'Unknown',
    'Does the word "deed" mean the same thing in sentences "He signed the deed." and "I inherited the deed to the house."?': 'Unknown',
    'Does the word "question" mean the same thing in sentences "His claim to the property has come under question." and "He obeyed without question."?': 'Unknown',
    'Does the word "distribute" mean the same thing in sentences "The publisher wants to distribute the book in Asia." and "The function distributes the values evenly."?': 'Unknown',
    'Does the word "return" mean the same thing in sentences "Return her love." and "Return a compliment."?': 'Unknown',
    'Does the word "officer" mean the same thing in sentences "He is an officer of the court." and "The club elected its officers for the coming year."?': 'Unknown',
    'Does the word "clutch" mean the same thing in sentences "To clutch power." and "She clutched her purse."?': 'Unknown',
    'Does the word "dissipation" mean the same thing in sentences "The dissipation of the mist." and "Mindless dissipation of natural resources."?': 'Unknown',
    'Does the word "portfolio" mean the same thing in sentences "He remembered her because she was carrying a large portfolio." and "He holds the portfolio for foreign affairs."?': 'Unknown',
    'Does the word "play" mean the same thing in sentences "They gave full play to the artist\'s talent." and "It was all done in play."?': 'Unknown',
    'Does the word "brush" mean the same thing in sentences "She gave her hair a quick brush." and "The dentist recommended two brushes a day."?': 'Unknown',
    'Does the word "rise" mean the same thing in sentences "They asked for a 10% rise in rates." and "The rising of the Holy Ghost."?': 'Unknown',
    'Does the word "studio" mean the same thing in sentences "His studio was cramped when he began as an artist." and "You don\'t need a studio to make a passport photograph."?': 'Unknown',
    'Does the word "cushion" mean the same thing in sentences "To cushion a blow." and "Cushion the blow."?': 'Unknown',
    'Does the word "plenty" mean the same thing in sentences "There was plenty of food for everyone." and "It must have cost plenty."?': 'Unknown',
    'Does the word "look" mean the same thing in sentences "A look of triumph." and "His look was fixed on her eyes."?': 'Unknown',
    'Does the word "recrudescence" mean the same thing in sentences "A recrudescence of racism." and "A recrudescence of the symptoms."?': 'Unknown',
    'Does the word "metadata" mean the same thing in sentences "Most websites contain metadata to tell the computer how to lay the words out on the screen." and "A library catalog is metadata because it describes publications."?': 'Unknown',
    'Does the word "streak" mean the same thing in sentences "A streak of wildness." and "He has a stubborn streak."?': 'Unknown',
    'Does the word "elapse" mean the same thing in sentences "He allowed a month to elapse before beginning the work." and "Several days elapsed before they met again."?': 'Unknown',
    'Does the word "corner" mean the same thing in sentences "He tripled to the rightfield corner." and "The southeastern corner of the Mediterranean."?': 'Unknown'

}

selected_questions = human_readable_questions_short
with_reasoning = ""
explain = True
if explain:
    with_reasoning = ' with reasoning'

print(f'Answer all {len(selected_questions)} questions with Yes or No{with_reasoning}!')
print(*human_readable_questions_short.keys(), sep='\n')
