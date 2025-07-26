import re

class SentenceNormalizer:
    def __init__(self):
        pass

    def makeSentenceHumanReadable(self, sentence: str, version: str = 'new') -> str:
        if version == 'old':
            return self.makeSentenceHumanReadableOld(sentence)
        else:
            return self.makeSentenceHumanReadableNew(sentence)



    def makeSentenceHumanReadableOld(self, sentence: str) -> str:
        """
        Replaces contractions for better readability both for humans and for chatbots.
        Original implementation with explicit replacements.
        """
        # Contractions
        sentence = sentence.replace(" 's", "\\'s")
        sentence = sentence.replace(" 't", "\\'t")
        sentence = sentence.replace(" 'd", "\\'d")
        sentence = sentence.replace(" 'm", "\\'m")
        sentence = sentence.replace(" 've", "\\'ve")
        sentence = sentence.replace(" 'll", "\\'ll")
        sentence = sentence.replace(" 're", "\\'re")
        sentence = sentence.replace(" n't", "n\\'t")
        sentence = sentence.replace("o'clock", "o\\'clock")

        # Punctuation
        sentence = sentence.replace(" ,", ",")
        sentence = sentence.replace(" .", ".")
        sentence = sentence.replace(" ?", "?")
        sentence = sentence.replace(" !", "!")

        # Special uppercase cases
        sentence = sentence.replace("Do n't", "Don\\'t")
        sentence = sentence.replace("Wo n't", "Won\\'t")
        sentence = sentence.replace("Ca n't", "Can\\'t")

        # Special lowercase cases
        sentence = sentence.replace("do n't", "don\\'t")
        sentence = sentence.replace("wo n't", "won\\'t")
        sentence = sentence.replace("ca n't", "can\\'t")

        # General n't replacement
        sentence = sentence.replace(" n't", "not")

        return sentence


    def makeSentenceHumanReadableNew(self, sentence: str) -> str:
        """
        Replaces contractions for better readability using a more automated, concise approach.
        New implementation using regex patterns and dictionaries.
        """
        # Contractions mapping
        contractions = {
            r" 's": "\\'s",
            r" 't": "\\'t",
            r" 'd": "\\'d",
            r" 'm": "\\'m",
            r" 've": "\\'ve",
            r" 'll": "\\'ll",
            r" 're": "\\'re",
            r" n't": "n\\'t",
            r"\bo'(?=\w)": r"o\\'"
        }

        # Special cases for n't
        special_contractions = {
            r"Do n't": "Don\\'t",
            r"Wo n't": "Won\\'t",
            r"Ca n't": "Can\\'t",
            r"do n't": "don\\'t",
            r"wo n't": "won\\'t",
            r"ca n't": "can\\'t",
        }

        # Punctuation
        punctuation = r" ([,.?!])"

        # Apply all replacements
        for pattern, replacement in contractions.items():
            sentence = re.sub(pattern, replacement, sentence)

        for pattern, replacement in special_contractions.items():
            sentence = sentence.replace(pattern, replacement)

        # Fix punctuation
        sentence = re.sub(punctuation, r"\1", sentence)

        # General n't replacement (after the special cases)
        sentence = sentence.replace(" n't", "not")

        return sentence


def main():
    sn = SentenceNormalizer()
    test_sentence = (r"My boss is out on another of his three martini lunches ."
                     r" Will you join us at six o'clock for martinis ? We 've been swimming for hours just to get to the other side ."
                     r" A big fish was swimming in the tank . Do n't fire until you see the whites of their eyes . The gun fired .")

    print("Using NEW algorithm (default):")
    print(sn.makeSentenceHumanReadable(test_sentence))

    print("\nUsing OLD algorithm:")
    print(sn.makeSentenceHumanReadable(test_sentence, version="old"))


if __name__ == '__main__':
    main()