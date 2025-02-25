def make_sentence_human_readable(sentence):
        """Replaces contractions for better readability both for humans and for chatbots."""
        sentence = sentence.replace(" 's", "'s")
        sentence = sentence.replace(" .", ".")
        sentence = sentence.replace("n't", "not")
        return sentence
