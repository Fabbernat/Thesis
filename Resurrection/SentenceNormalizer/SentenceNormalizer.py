import re

class SentenceNormalizer:
    def __init__(self):
        pass

    def makeSentenceHumanReadable(self, sentence: str) -> str:
        """
        Replaces contractions for better readability both for humans and for chatbots.
        """
        # Escape all apostrophes
        sentence = sentence.replace("'", "\\'")

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

        return sentence


def testMakeSentenceHumanReadable():
    sn = SentenceNormalizer()
    test_sentence = ("My boss is out on another of his three martini lunches ."
                     " Will you join us at six o'clock for martinis ? We 've been swimming for hours just to get to the other side ."
                     " A big fish was swimming in the tank . Do n't fire until you see the whites of their eyes . The gun fired .")

    print("Using the NEW algorithm (default):")
    print(sn.makeSentenceHumanReadable(test_sentence))



if __name__ == '__main__':
    testMakeSentenceHumanReadable()