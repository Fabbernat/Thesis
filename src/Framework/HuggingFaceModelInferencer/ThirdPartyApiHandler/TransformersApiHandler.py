try:
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.google.google import tokenizeAutoModelForGoogle0
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.qwen.qwen import tokenizeAutoModelForQwenAndSimilar0, \
        generateIds1, convertIds2
    from src.Framework.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString
    from src.Framework.HuggingFaceModelInferencer.modelname import MODEL_NAME
except Exception as e:
    from .orgs.google import tokenizeAutoModelForGoogle0
    from .orgs.qwen import  tokenizeAutoModelForQwenAndSimilar0
    from MessagesAsASingleStringBuilder.Builder import getMessagesAsString
    from modelname import MODEL_NAME

from transformers import AutoTokenizer
from pathlib import Path

class TransformersApiHandler:
    def __init__(self):
        print('TransformersApiHandler() initalized')
        self.tokenizer = object
        self.model = object
        self.generatedIds = object
        self.response = object
        self.modelInputs = object

    def DoAutotokenizerFromPretrained(self):
        print("Doing AutoTokenizer.from_pretrained...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME)  # should be of type TextKwargs(), but TextKwargs is inaccessible from here for some reason
        print("AutoTokenizer.from_pretrained done!")

    # for google models
    def google(self):
        print('google path chosen')
        tokenizeAutoModelForGoogle0()

    def qwen(self):
        print('qwen path chosen')
        self.model, self.modelInputs = tokenizeAutoModelForQwenAndSimilar0(self.model, self.tokenizer)
        generatedIds = generateIds1(self.model, self.modelInputs)
        convertedIds = convertIds2(self.modelInputs, self.generatedIds)

        return generatedIds, convertedIds

    def microsoft(self):
        pass


    def batchDecodeGenerateFinalAnswer3(self, convertedTensors):
        if self.tokenizer is None:
            print('failed to give self.tokenizer a value using AutoTokenizer.from_pretrained(MODEL_NAME.strip()).')
        try:
            print(convertedTensors)
            self.response = self.tokenizer.batch_decode(convertedTensors, skip_special_tokens=True)  # [NUMBER_OF_DESIRED_ANSWERS] # [0] makes the answers longer for some reason, so [NUMBER_OF_DESIRED_ANSWERS] is not needed.
        except AttributeError as ae:
            print('AttributeError trying to batch_decode generatedIds:', ae)
        except TypeError as te:
            print('TypeError trying to batch_decode generatedIds:', te)
        return self.response, self.generatedIds, self.tokenizer  # need to test all three

    def decodeOutputsSkippingSpecialTokens(self):
        try:
            self.tokenizer.decode(self.generatedIds, skip_special_tokens=True)
        except AttributeError as ae:
            raise Exception("AttributeError while trying to decode outputs skipping special tokens.", ae)