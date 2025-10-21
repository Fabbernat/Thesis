from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.errors import HFValidationError

try:
    from MessagesAsASingleStringBuilder.Builder import getMessagesAsString
except ImportError as ie:
    print("Could not import getMessagesAsString: ", ie)
except Exception as e:
    print("Exception occured: ", e)


class TransformersApiHandler:
    def __init__(self):
        print('TransformersApiHandler() started')
        self.tokenizer = object # should be of type TextKwargs(), but TextKwargs is inaccessible from here for some reason
        self.model = object
        self.generated_ids = object
        self.response = object
        self.modelInputs = object

    def tokenizeAutoModelForQwenAndSimilar0(self):
        from Config.Config import MODEL_NAME
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

        promptAsText = self.tokenizer.apply_chat_template(
            getMessagesAsString(10),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # For Qwen3-32B
        )

        self.modelInputs = self.tokenizer([promptAsText], return_tensors="pt").to(self.model.device)


    # for gemma models
    def tokeniteAutoModelForGoogle0(self):
        from transformers import pipeline
        from src.Framework.HuggingFaceModelInferencer.Config.Config import MODEL_NAME
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            device="cuda",
        )
        text = getMessagesAsString(10)
        outputs = pipe(text, max_new_tokens=256)
        response = outputs[0]["generated_text"]

        return response


    def generateIds1(self):
        self.generated_ids = self.model.generate(**self.modelInputs, max_new_tokens=32768)
        return self.generated_ids

    def convertIds2(self):
        self.generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(self.modelInputs.input_ids, self.generated_ids)
        ]

        return self.generated_ids

    def generateFinalAnswer3(self):
        if self.tokenizer is None:
            print('failed to give self.tokenizer a value using AutoTokenizer.from_pretrained(MODEL_NAME.strip()).')
        try:
            self.response = self.tokenizer.batch_decode(self.generated_ids, skip_special_tokens=True)[0] # [NUMBER_OF_DESIRED_ANSWERS] # [0] makes the answers longer for some reason, so [NUMBER_OF_DESIRED_ANSWERS] is not needed.
            print(self.generated_ids)
        except AttributeError as ae:
            print('AttributeError trying to batch_decode generated_ids:', ae)
        except TypeError as te:
            print('TypeError trying to batch_decode generated_ids:', te)
        return self.response, self.generated_ids, self.tokenizer # need to test all three
    def decodeOutputsSkippingSpecialTokens(self):
        try:
            self.tokenizer.decode(self.generated_ids, skip_special_tokens=True)
        except AttributeError as ae:
            raise Exception("AttributeError while trying to decode outputs skipping special tokens.", ae)