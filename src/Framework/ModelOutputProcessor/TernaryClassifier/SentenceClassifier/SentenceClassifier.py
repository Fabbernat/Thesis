def classifySentence(linebreaklessSentence: str, phrases=False) -> str:
    """
    function that gets a linebreaklessSentence as input and may output 3 different characters based on the sentence: - 'T' if the linebreaklessSentence contains the word "Yes" (case-sensitive), or an affirmative message. - 'F' if the linebreaklessSentence contains the word "No" (case-sensitive), or a not affirmative message. - '?' in any other cases, where the intent of the sentence is unclear.
    Ez a legjobb ötletem a modell biasának az eldöntésére, de ez biztosan nem osztályozza be a szándékokat 100%-os pontossággal
    :return:
    """
    #TODO mi van ha "Yes and No" a válasz, vagy "eyes"?
    sentence = linebreaklessSentence.strip()
    if sentence.upper() == 'T' or sentence.lower() == 'Yes' or sentence.lower() == 'Yes.' or 'Yes' in sentence:
        return 'T'
    if sentence.upper() == 'F' or sentence.lower() == 'No' or sentence.lower() == 'No.' or 'No' in sentence:
        return 'F'

    if phrases:
        return classifyByPhrases(sentence)
    else:
        return classifyByKeywords(sentence)

def classifyByPhrases(sentence: str) -> str:
    from src.Framework.ModelOutputProcessor.config import AFFIRMATIVE_PHRASES, NEGATIVE_PHRASES

    affirmativePhrases = AFFIRMATIVE_PHRASES
    negativePhrases = NEGATIVE_PHRASES

    if any(phrase.lower() in sentence.lower() for phrase in affirmativePhrases):
        return 'T'
    if any(phrase.lower() in sentence.lower() for phrase in negativePhrases):
        return 'F'
    return '?'

def classifyByKeywords(sentence: str) -> str:
    from src.Framework.ModelOutputProcessor.config import AFFIRMATIVE_KEYWORDS, NEGATIVE_KEYWORDS
    affirmativeKeywords = AFFIRMATIVE_KEYWORDS
    negativeKeywords = NEGATIVE_KEYWORDS

    if any(word.lower() in sentence.lower() for word in affirmativeKeywords):
        return 'T'
    if any(word.lower() in sentence.lower() for word in negativeKeywords):
        return 'F'
    return '?'
