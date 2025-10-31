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
