def classifySentence(linebreaklessSentence: str, phrases=False) -> str:
    """
    En:
    function that gets a linebreaklessSentence as input and may output 3 different characters based on the sentence: - 'T' if the linebreaklessSentence contains the word "Yes" (case-sensitive), or an affirmative message. - 'F' if the linebreaklessSentence contains the word "No" (case-sensitive), or a not affirmative message. - '?' in any other cases, where the intent of the sentence is unclear.

    Hu:
    Fogadjuk el a "Yes", vagy "No" szó előfordulását (vagy True/False) valid válasznak. Ez az esetek 99%-ában helyesen ítéli meg a modell biasát. Nyilván ha a modell más szavakkal írja le, vagy más nyelven válaszol, akkor összetettebb check kell.
    :return:
    """

    sentence = linebreaklessSentence.strip()
    if sentence.upper() == 'T' or sentence.lower() == 'true' or sentence.lower() == 'yes' or sentence.lower() == 'yes.' or 'Yes' in sentence:
        return 'T'
    if sentence.upper() == 'F' or sentence.lower() == 'false' or sentence.lower() == 'no' or sentence.lower() == 'no.' or 'No' in sentence:
        return 'F'

    return '?'