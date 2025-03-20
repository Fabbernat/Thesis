import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import re

data = [

        """A bartender is working at a saloon, serving drinks to customers. After he fills a stereotypically Irish man's bucket with beer, Carrie Nation and her followers burst inside. They assault the Irish man, pulling his hat over his eyes and then dumping the beer over his head. The group then begin wrecking the bar, smashing the fixtures, mirrors, and breaking the cash register. The bartender then sprays seltzer water in Nation's face before a group of policemen appear and order everybody to leave.[1]""",


        "The moon, painted with a smiling face hangs over a park at night. A young couple walking past a fence learn on a railing and look up. The moon smiles. They embrace, and the moon's smile gets bigger. They then sit down on a bench by a tree. The moon's view is blocked, causing him to frown. In the last scene, the man fans the woman with his hat because the moon has left the sky and is perched over her shoulder to see everything better.""",


        """The film, just over a minute long, is composed of two shots. In the first, a girl sits at the base of an altar or tomb, her face hidden from the camera. At the center of the altar, a viewing portal displays the portraits of three U.S. Presidents—Abraham Lincoln, James A. Garfield, and William McKinley—each victims of assassination. In the second shot, which runs just over eight seconds long, an assassin kneels feet of Lady Justice.""",


    """Lasting just 61 seconds and consisting of two shots, the first shot is set in a wood during winter. The actor representing then vice-president Theodore Roosevelt enthusiastically hurries down a hillside towards a tree in the foreground. He falls once, but rights himself and cocks his rifle. Two other men, bearing signs reading ""His Photographer"" and ""His Press Agent"" respectively, follow him into the shot; the photographer sets up his camera. ""Teddy"" aims his rifle upward at the tree and fells what appears to be a common house cat, which he then proceeds to stab. ""Teddy"" holds his prize aloft, and the press agent takes notes. The second shot is taken in a slightly different part of the wood, on a path. ""Teddy"" rides the path on his horse towards the camera and out to the left of the shot, followed closely by the press agent and photographer, still dutifully holding their signs.""",

    """The earliest known adaptation of the classic fairytale, this films shows Jack trading his cow for the beans, his mother forcing him to drop them in the front yard, and beig forced upstairs. As he sleeps, Jack is visited by a fairy who shows him glimpses of what will await him when he ascends the bean stalk. In this version, Jack is the son of a deposed king. When Jack wakes up, he finds the beanstalk has grown and he climbs to the top where he enters the giant's home. The giant finds Jack, who narrowly escapes. The giant chases Jack down the bean stalk, but Jack is able to cut it down before the giant can get to safety. He falls and is killed as Jack celebrates. The fairy then reveals that Jack may return home as a prince.""",
    """Alice follows a large white rabbit down a ""Rabbit-hole"". She finds a tiny door. When she finds a bottle labeled ""Drink me"", she does, and shrinks, but not enough to pass through the door. She then eats something labeled ""Eat me"" and grows larger. She finds a fan when enables her to shrink enough to get into the ""Garden"" and try to get a ""Dog"" to play with her. She enters the ""White Rabbit's tiny House,"" but suddenly resumes her normal size. In order to get out, she has to use the ""magic fan."" She enters a kitchen, in which there is a cook and a woman holding a baby. She persuades the woman to give her the child and takes the infant outside after the cook starts throwing things around. The baby then turns into a pig and squirms out of her grip. ""The Duchess's Cheshire Cat"" appears and disappears a couple of times to Alice and directs her to the Mad Hatter's ""Mad Tea-Party."" After a while, she leaves. The Queen invites Alice to join the ""ROYAL PROCESSION"": a parade of marching playing cards and others headed by the White Rabbit. When Alice ""unintentionally offends the Queen"", the latter summons the ""Executioner"". Alice ""boxes the ears"", then flees when all the playing cards come for her. Then she wakes up and realizes it was all a dream.""",
    """The film opens with two bandits breaking into a railroad telegraph office, where they force the operator at gunpoint to have a train stopped and to transmit orders for the engineer to fill the locomotive's tender at the station's water tank. They then knock the operator out and tie him up. As the train stops it is boarded by the bandits‍—‌now four. Two bandits enter an express car, kill a messenger and open a box of valuables with dynamite; the others kill the fireman and force the engineer to halt the train and disconnect the locomotive. The bandits then force the passengers off the train and rifle them for their belongings. One passenger tries to escape but is instantly shot down. Carrying their loot, the bandits escape in the locomotive, later stopping in a valley where their horses had been left. Meanwhile, back in the telegraph office, the bound operator awakens, but he collapses again. His daughter arrives bringing him his meal and cuts him free, and restores him to consciousness by dousing him with water. There is some comic relief at a dance hall, where an Eastern stranger is forced to dance while the locals fire at his feet. The door suddenly opens and the telegraph operator rushes in to tell them of the robbery. The men quickly form a posse, which overtakes the bandits, and in a final shootout kills them all and recovers the stolen mail.""",
    """The film is about a family who move to the suburbs, hoping for a quiet life. Things start to go wrong, and the wife gets violent and starts throwing crockery, leading to her arrest.""",
    """The opening scene shows the interior of the robbers' den. The walls are decorated with the portraits of notorious criminals and pictures illustrating the exploits of famous bandits. Some of the gang are lounging about, while others are reading novels and illustrated papers. Although of youthful appearance, each is dressed like a typical Western desperado. The ""Bandit Queen,"" leading a blindfolded new recruit, now enters the room. He is led to the center of the room, raises his right hand and is solemnly sworn in. When the bandage is removed from his eyes he finds himself looking into the muzzles of a dozen or more 45's. The gang then congratulates the new member and heartily shake his hand. The ""Bandit Queen"" who is evidently the leader of the gang, now calls for volunteers to hold up a train. All respond, but she picks out seven for the job who immediately leave the cabin. The next scene shows the gang breaking into a barn. They steal ponies and ride away. Upon reaching the place agreed upon they picket their ponies and leaving them in charge of a trusted member proceed to a wild mountain spot in a bend of the railroad, where the road runs over a steep embankment. The spot is an ideal one for holding up a train. Cross ties are now placed on the railroad track and the gang hide in some bushes close by and wait for the train. The train soon approaches and is brought to a stop. The engineer leaves his engine and proceeds to remove the obstruction on the track. While he is bending over one of the gang sneaks up behind them and hits him on the head with an axe, and knocks him senseless down the embankment, while the gang surround the train and hold up the passengers. After securing all the ""valuables,"" consisting principally of candy and dolls, the robbers uncouple the engine and one car and make their escape just in time to avoid a posse of police who appear on the scene. Further up the road they abandon the engine and car, take to the woods and soon reach their ponies. In the meantime the police have learned the particulars of the hold-up from the frightened passengers and have started up the railroad tracks after the fleeing robbers. The robbers are next seen riding up the bed of a shallow stream and finally reach their den, where the remainder of the gang have been waiting for them. Believing they have successfully eluded their pursuers, they proceed to divide the ""plunder."" The police, however, have struck the right trail and are in close pursuit. While the ""plunder"" is being divided a sentry gives the alarm and the entire gang, abandoning everything, rush from the cabin barely in time to escape capture. The police make a hurried search and again start in pursuit. The robbers are so hard pressed that they are unable to reach their ponies, and are obliged to take chances on foot. The police now get in sight of the fleeing robbers and a lively chase follows through tall weeds, over a bridge and up a steep hill. Reaching a pond the police are close on their heels. The foremost robbers jump in clothes and all and strike out for the opposite bank. Two hesitate and are captured. Boats are secured and after an exciting tussle the entire gang is rounded up. In the mix up one of the police is dragged overboard. The final scene shows the entire gang of bedraggled and crestfallen robbers tied together with a rope and being led away by the police. Two of the police are loaded down with revolvers, knives and cartridge belts, and resemble walking aresenals. As a fitting climax a confederate steals out of the woods, cuts the rope and gallantly rescues the ""Bandit Queen.""",
    """Scenes are introduced using lines of the poem.[2] Santa Claus, played by Harry Eytinge, is shown feeding real reindeer[4] and finishes his work in the workshop. Meanwhile, the children of a city household hang their stockings and go to bed, but unable to sleep they engage in a pillow fight. Santa Claus leaves his home on a sleigh with his reindeer. He enters the children's house through the chimney, and leaves the presents. The children come down the stairs and enjoy their presents.""",
    """The Rarebit Fiend gorges on Welsh rarebit at a restaurant. When he leaves, he begins to get dizzy as he starts to hallucinate. He desperately tries to hang onto a lamppost as the world spins all around him. A man helps him get home. He falls into bed and begins having more hallucinatory dreams. During a dream sequence, the furniture begins moving around the room. Imps emerge from a floating Welsh rarebit container and begin poking his head as he sleeps. His bed then begins dancing and spinning wildly around the room before flying out the window with the Fiend in it. The bed floats across the city as the Fiend floats up and off the bed. He hangs off the back and eventually gets caught on a weathervane atop a steeple. His bedclothes tear and he falls from the sky, crashing through his bedroom ceiling. The Fiend awakens from the dream after falling out of his bed.""",
    """The film features a train traveling through the Rockies and a hold up created by two thugs placing logs on the line. They systematically rob the wealthy occupants at gunpoint and then make their getaway along the tracks and later by a hi-jacked horse and cart."""
]

train_years = [1901, 1901, 1901, 1901, 1902, 1903, 1903, 1904, 1905, 1905, 1906, 1906]

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[\d+\]', '', text) # Remove citation markers like [1]
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    return text

# Preprocess the text data
processed_data = [preprocess_text(text) for text in data]

# Encode the labels (years) - although in this case, years are already numerical, encoding might be useful if you were dealing with categorical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(train_years)

# Split data into training and testing sets (for final evaluation - optional for now, focus on cross-validation)
# X_train_text, X_test_text, y_train, y_test = train_test_split(processed_data, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels)
X_train_text = processed_data
y_train = encoded_labels
X_test_text = None # Set test data to None as we are focusing on cross-validation on the entire dataset for now
y_test = None

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english') # Added stop_words and ngram_range

# Logistic Regression Classifier
logreg_classifier = LogisticRegression(random_state=42, solver='liblinear', C=1.0) # Added solver and C for potential tuning

# RandomForest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10) # Example RandomForest - parameters can be tuned

# Pipeline for TF-IDF and Logistic Regression
pipeline_logreg = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('clf', logreg_classifier)
])

# Pipeline for TF-IDF and Random Forest
pipeline_rf = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('clf', rf_classifier)
])

# --- Model Evaluation using Cross-Validation ---
print("Cross-validation scores (Logistic Regression):")
cv_scores_logreg = cross_val_score(pipeline_logreg, X_train_text, y_train, cv=StratifiedKFold(n_splits=5)) # StratifiedKFold for class balance
cv_scores_rf = cross_val_score(pipeline_rf, X_train_text, y_train, cv=StratifiedKFold(n_splits=5)) # StratifiedKFold for class balance
print(cv_scores_logreg)
print(f"Mean CV Accuracy (Logistic Regression): {np.mean(cv_scores_logreg):.4f}")

print("\nCross-validation scores (Random Forest):")
cv_scores_rf = cross_val_score(pipeline_rf, X_train_text, y_train, cv=StratifiedKFold(n_splits=5)) # StratifiedKFold for class balance
print(cv_scores_rf)
print(f"Mean CV Accuracy (Random Forest): {np.mean(cv_scores_rf):.4f}")

# --- Model Training on Full Training Data ---
pipeline_logreg.fit(X_train_text, y_train)
pipeline_rf.fit(X_train_text, y_train)

# --- Model Evaluation on Test Data --- (Optional, but good practice)
if X_test_text is not None: # Check if test data exists before evaluating
    y_pred_logreg = pipeline_logreg.predict(X_test_text)
    y_pred_rf = pipeline_rf.predict(X_test_text)

    print("\n--- Logistic Regression Performance on Test Set ---")
    print(f"Accuracy (Logistic Regression): {accuracy_score(y_test, y_pred_logreg):.4f}")
    print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_logreg, target_names=label_encoder.inverse_transform(np.unique(y_train)))) # Added target_names

    print("\n--- Random Forest Performance on Test Set ---")
    print(f"Accuracy (Random Forest): {accuracy_score(y_test, y_pred_rf):.4f}")
    print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf, target_names=label_encoder.inverse_transform(np.unique(y_train)))) # Added target_names


print("\n--- Suggestions for Improvement ---")
print("- **Hyperparameter Tuning:**  Experiment with different parameters for TF-IDF (e.g., `ngram_range`, `max_df`, `min_df`) and the classifiers (Logistic Regression: `C`, `penalty`; RandomForest: `n_estimators`, `max_depth`, `min_samples_split`). Use GridSearchCV or RandomizedSearchCV for systematic tuning.")
print("- **Feature Engineering:** Consider adding more features.  For example:")
print("    - **Text Length:**  Longer descriptions might correlate with certain years.")
print("    - **Word Count:** Similar to text length.")
print("    - **Presence of Specific Keywords:**  Certain words might be more common in descriptions from specific eras.")
print("- **More Data:**  The dataset is very small. Increasing the training data significantly would likely improve performance.")
print("- **Different Models:** Explore other classification models like Support Vector Machines (SVM), Naive Bayes, or Gradient Boosting algorithms (e.g., XGBoost, LightGBM).")
print("- **Stemming/Lemmatization:** Apply stemming or lemmatization during preprocessing to reduce words to their root form, which can improve TF-IDF effectiveness.")
print("- **Handling Class Imbalance:** If some years are much more frequent than others in a larger dataset, consider techniques to handle class imbalance (e.g., SMOTE, class weights in classifiers).")
print("- **Error Analysis:** Look at the classification report and identify which years are being misclassified most often. Analyze the descriptions of those misclassified examples to understand why and potentially improve preprocessing or features.")