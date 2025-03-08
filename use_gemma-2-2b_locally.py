import torch
from transformers import pipeline

# Windows
model_path = r'C:\codes\gemma22b\gemma-2-2b-it'

pipe = pipeline(
    'text-generation',
    model=model_path,
    model_kwargs={'torch_dtype':torch.bfloat16},
    device='cuda', # replace with 'mps' on a Mac device
)

question = '''
Answer all 60 questions with Yes or No with reasoning!
Does the word "defeat" mean the same thing in sentences "It was a narrow defeat." and "The army's only defeat."?
Does the word "groom" mean the same thing in sentences "Groom the dogs." and "Sheila groomed the horse."?
Does the word "penetration" mean the same thing in sentences "The penetration of upper management by women." and "Any penetration, however slight, is sufficient to complete the offense."?
Does the word "hit" mean the same thing in sentences "We hit Detroit at one in the morning but kept driving through the night." and "An interesting idea hit her."?
Does the word "deliberation" mean the same thing in sentences "He was a man of judicial deliberation." and "A little deliberation would have deterred them."?
Does the word "navel" mean the same thing in sentences "They argued whether or not Adam had a navel." and "You were not supposed to show your navel on television."?
Does the word "afforest" mean the same thing in sentences "After we leave the quarry, we intend to afforest the land and turn it into a nature reserve." and "Afforest the mountains."?
Does the word "solve" mean the same thing in sentences "Solve an old debt." and "Did you solve the problem?"?
Does the word "purchase" mean the same thing in sentences "They offer a free hamburger with the purchase of a drink." and "They closed the purchase with a handshake."?
Does the word "software" mean the same thing in sentences "Did you test the software package to ensure completeness?" and "The market for software is expected to expand."?
Does the word "push" mean the same thing in sentences "Some details got lost in the push to get the project done." and "The army made a push toward the sea."?
Does the word "bake" mean the same thing in sentences "Idaho potatoes bake beautifully." and "This oven bakes potatoes."?
Does the word "relieve" mean the same thing in sentences "Relieve the pressure and the stress." and "This pill will relieve your headaches."?
Does the word "style" mean the same thing in sentences "In the characteristic New York style." and "This style of shoe is in demand."?
Does the word "crumb" mean the same thing in sentences "Crumb the table." and "Crumb a cutlet."?
Does the word "include" mean the same thing in sentences "I include you in the list of culprits." and "The list includes the names of many famous writers."?
Does the word "companion" mean the same thing in sentences "His dog has been his trusted companion for the last five years." and "Drinking companions."?
Does the word "reveal" mean the same thing in sentences "The actress won't reveal how old she is." and "He revealed the children found."?
Does the word "presence" mean the same thing in sentences "I'm convinced that there was a presence in that building that I can't explain, which led to my heroic actions." and "She blushed in his presence."?
Does the word "relax" mean the same thing in sentences "Don't relax your efforts now." and "The rules relaxed after the new director arrived."?
Does the word "parity" mean the same thing in sentences "Parity is often used to check the integrity of transmitted data." and "The parity of the mother must be considered."?
Does the word "raise" mean the same thing in sentences "To raise a wall, or a heap of stones." and "Raise a barn."?
Does the word "suspend" mean the same thing in sentences "Suspend the particles." and "The prison sentence was suspended."?
Does the word "amass" mean the same thing in sentences "To amass a treasure or a fortune." and "She is amassing a lot of data for her thesis."?
Does the word "term" mean the same thing in sentences "A healthy baby born at full term." and "He learned many medical terms."?
Does the word "leash" mean the same thing in sentences "He's always gotten a long leash." and "Kept a tight leash on his emotions."?
Does the word "conversion" mean the same thing in sentences "The conversion of equations." and "His conversion to the Catholic faith."?
Does the word "making" mean the same thing in sentences "The making of measurements." and "It was already in the making."?
Does the word "set" mean the same thing in sentences "Before the set of sun." and "They played two sets of tennis after dinner."?
Does the word "mate" mean the same thing in sentences "He lost the mate to his shoe." and "Camels hate leaving their mates."?
Does the word "expression" mean the same thing in sentences "They stared at the newcomer with a puzzled expression." and "His manner of expression showed how much he cared."?
Does the word "rim" mean the same thing in sentences "Rim a hat." and "Sugar rimmed the dessert plate."?
Does the word "cure" mean the same thing in sentences "Cure meats." and "Cure hay"?
Does the word "rift" mean the same thing in sentences "My marriage is in trouble, the fight created a rift between us and we can't reconnect." and "The Grand Canyon is a rift in the Earth's surface, but is smaller than some of the undersea ones."?
Does the word "swim" mean the same thing in sentences "We had to swim for 20 minutes to reach the shore." and "A big fish was swimming in the tank."?
Does the word "quiet" mean the same thing in sentences "The library was so quiet you could hear a pin drop." and "She tried to quiet the crying baby."?
Does the word "top" mean the same thing in sentences "She reached the top of the mountain." and "He spun the wooden top on the table."?
Does the word "consultation" mean the same thing in sentences "A consultation of several medical specialists." and "Frequent consultations with his lawyer."?
Does the word "chiromance" mean the same thing in sentences "She refused to chiromance my fate." and "The Gypsies chiromanced."?
Does the word "bank" mean the same thing in sentences "Bank on your good education." and "The pilot had to bank the aircraft."?
Does the word "rag" mean the same thing in sentences "Rag that old tune." and "Rag ore."?
Does the word "work" mean the same thing in sentences "Work equals force times distance." and "Work is done against friction to drag a bag along the ground."?
Does the word "allowance" mean the same thing in sentences "He objected to the allowance of smoking in the dining room." and "A child's allowance should not be too generous."?
Does the word "contact" mean the same thing in sentences "Litmus paper turns red on contact with an acid." and "He used his business contacts to get an introduction to the governor."?
Does the word "virus" mean the same thing in sentences "The virus of jealousy is latent in everyone." and "He caught a virus and had to stay home from school."?
Does the word "humour" mean the same thing in sentences "The sensitive subject was treated with humour, but in such way that no one was offended." and "She has a great sense of humour, and I always laugh a lot whenever we get together."?
Does the word "neighbor" mean the same thing in sentences "What is the closest neighbor to the Earth?" and "Fort Worth is a neighbor of Dallas."?
Does the word "sinking" mean the same thing in sentences "He could not control the sinking of his legs." and "After several hours of sinking an unexpected rally rescued the market."?
Does the word "sneak" mean the same thing in sentences "Sneak a look." and "Sneak a cigarette."?
Does the word "fix" mean the same thing in sentences "Fix your eyes on this spot." and "Fix a race."?
Does the word "impulse" mean the same thing in sentences "The impulse knocked him over." and "The total impulse from the impact will depend on the kinetic energy of the bullet."?
Does the word "fetish" mean the same thing in sentences "I know a guy who has a foot fetish." and "Common male fetishes are breasts , legs , hair , shoes , and underwear."?
Does the word "seizure" mean the same thing in sentences "The seizure of a thief , a property , a throne , etc ." and "The search warrant permitted the seizure of evidence ."?
Does the word "pattern" mean the same thing in sentences "The American constitution has provided a pattern for many republics ." and "They changed their dietary pattern ."?
Does the word "conscience" mean the same thing in sentences "A person of unflagging conscience ." and "He has no conscience about his cruelty ."?
Does the word "demistify" mean the same thing in sentences "The article was written to demystify the mechanics of the internal combustion engine ." and "Let's demystify the event by explaining what it is all about ."?
Does the word "team" mean the same thing in sentences "We need more volunteers for the netball team." and "The IT manager leads a team of three software developers."?
Does the word "conduct" mean the same thing in sentences "You can not conduct business like this." and "To conduct the affairs of a kingdom."?
Does the word "administer" mean the same thing in sentences "Administer critical remarks to everyone present." and "She administers the funds."?
Does the word "abort" mean the same thing in sentences "I wasted a year of my life working on an abort." and "He sent a short message requesting an abort due to extreme winds in the area."?
'''

messages = [{'role':'user', 'content': question}]

outputs = pipe(messages, max_new_tokens=512)
assistant_response = outputs[0]['generated_text'][-1]['content'].strip()
print(assistant_response)