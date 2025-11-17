from pathlib import Path

try:
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.google.google import tokenizeAutoModelForGoogle0
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.qwen.qwen import \
        tokenizeAutoModelForQwenAndSimilar0
    from src.Framework.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString
    from src.Framework.HuggingFaceModelInferencer.modelname import MODEL_NAME
    from src.Framework.HuggingFaceModelInferencer.config import NUMBER_OF_DESIRED_ANSWERS
except Exception as e:
    from .orgs.google import tokenizeAutoModelForGoogle0
    from .orgs.qwen import tokenizeAutoModelForQwenAndSimilar0
    from MessagesAsASingleStringBuilder.Builder import getMessagesAsString
    from modelname import MODEL_NAME
    from config import NUMBER_OF_DESIRED_ANSWERS

from transformers import AutoTokenizer


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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print('====== QWEN PATH STARTED ======')

        # 1) Load model
        print(f"Loading model: {MODEL_NAME}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            dtype="auto"
        )
        print("Model loaded:")
        print("  model class:", type(self.model))
        print("  device map:", self.model.hf_device_map)

        # 2) Load tokenizer
        print(f"Loading tokenizer: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("Tokenizer loaded:", self.tokenizer)

        # 3) Build prompt
        msgs = getMessagesAsString(NUMBER_OF_DESIRED_ANSWERS)
        print("MessagesAsString:", msgs)

        prompt = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        print("Prompt:")
        print(prompt)

        # 4) Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        print("Tokenized inputs:")
        for k, v in inputs.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        print("inputs object:", inputs)

        # 5) Generate
        print("Calling model.generate...")
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
        )

        print("Generate() output:")
        print("  output shape:", output.shape)
        print("  output device:", output.device)

        # 6) Slice prompt away
        input_len = inputs["input_ids"].shape[1]
        print("Prompt token length:", input_len)

        generated_ids = output[:, input_len:]
        print("Generated_ids sliced:")
        print("  shape:", generated_ids.shape)
        print("  tensor:", generated_ids)

        # 7) Decode
        decoded = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        print("Decoded output:")
        for i, d in enumerate(decoded):
            print(f"  decoded[{i}]: {repr(d)}")

        print("====== QWEN PATH END ======")

        return decoded, generated_ids

    def microsoft(self):
        pass

    def batchDecodeGenerateFinalAnswer(self, convertedTensors):
        if self.tokenizer is None:
            print('failed to give self.tokenizer a value using AutoTokenizer.from_pretrained(MODEL_NAME.strip()).')
        try:
            print(convertedTensors)
            self.response = self.tokenizer.batch_decode(convertedTensors,
                                                        skip_special_tokens=True)  # [NUMBER_OF_DESIRED_ANSWERS] # [0] makes the answers longer for some reason, so [NUMBER_OF_DESIRED_ANSWERS] is not needed.
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
