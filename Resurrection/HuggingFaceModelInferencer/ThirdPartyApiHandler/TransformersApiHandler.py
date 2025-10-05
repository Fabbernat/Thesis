from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.errors import HFValidationError


# from Resurrection.HuggingFaceModelInferencer.Config.Config import NUMBER_OF_DESIRED_ANSWERS

try:
    from Resurrection.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString
except Exception as e:
    print("Warning: Could not import getMessagesAsString:", e)
    getMessagesAsString = lambda: ""  # fallback: empty prompt


class TransformersApiHandler:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.generated_ids = None
        self.response = None
        self.inputs = None

    def tokenizeAutoModelForQwenAndSimilar0(self):
        from Resurrection.HuggingFaceModelInferencer.Config.Config import MODEL_NAME
        try:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto", device_map="auto") # torch_dtype is deprecated, but still necessary?


        except AttributeError as e:
            print("AttributeError in self.model.generate.", str(e))
        print(f'self.tokenizer before={self.tokenizer}')



        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        except HFValidationError as e:
            print("Invalid Hugging Face model name:", e)

        except Exception as e:
            print("Unexpected error while loading model:", e)

        print(f'self.tokenizer after={self.tokenizer}')

        prompt = self.tokenizer.apply_chat_template(getMessagesAsString(), tokenize=False, add_generation_prompt=True)

        self.inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)


    def tokeniteAutoModelForGoogle0(self):
        from transformers import pipeline
        from Resurrection.HuggingFaceModelInferencer.Config.Config import MODEL_NAME
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device="cuda",
        )
        text = getMessagesAsString()
        outputs = pipe(text, max_new_tokens=256)
        response = outputs[0]["generated_text"]

        return response

    def generateIds1(self):
        self.generated_ids = self.model.generate(**self.inputs, max_new_tokens=512)
        return self.generated_ids

    def convertIds2(self):
        self.generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(self.inputs.input_ids, self.generated_ids)
        ]

        return self.generated_ids

    def generateFinalAnswer3(self):
        if self.tokenizer is None:
            print('failed to give self.tokenizer a value using AutoTokenizer.from_pretrained(MODEL_NAME.strip()).')
        try:
            self.response = self.tokenizer.batch_decode(self.generated_ids, skip_special_tokens=True)[0] # [NUMBER_OF_DESIRED_ANSWERS] # [0] makes the answers longer for some reason, so [NUMBER_OF_DESIRED_ANSWERS] is not needed.
            print(self.generated_ids)
        except AttributeError as ae:
            print('failed to batch_decode generated_ids:', ae)
        return self.response, self.generated_ids, self.tokenizer # need to test all three

    def decodeOutputsSkippingSpecialTokens(self):
        try:
            self.tokenizer.decode(self.generated_ids, skip_special_tokens=True)
        except AttributeError as ae:
            raise Exception("AttributeError while trying to decode outputs skipping special tokens.", ae)