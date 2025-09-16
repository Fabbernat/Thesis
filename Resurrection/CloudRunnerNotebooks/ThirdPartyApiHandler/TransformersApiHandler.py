import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.errors import HFValidationError

from Resurrection.CloudRunnerNotebooks.MessagesAsASingleStringMaker.MessagesAsASingleStringMaker import getMessagesAsString


class TransformersApiHandler:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.outputs = None

        from Resurrection.CloudRunnerNotebooks.main import MODEL_NAME
        try:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        except AttributeError:
            sys.stderr.write("AttributeError in self.model.generate.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME.strip())
        except HFValidationError as e:
            print("Invalid Hugging Face model name:", e)
            return  # shut down gracefully
        except Exception as e:
            print("Unexpected error while loading model:", e)
            return


        prompt = tokenizer.apply_chat_template(getMessagesAsString(), tokenize=False)

        self.inputs = tokenizer(prompt, return_tensors="pt")

    def generateAnswers(self):
        self.outputs = self.model.generate(**self.inputs, max_new_tokens=50)

    def decodeOutputsSkippingSpecialTokens(self):
        try:
            self.tokenizer.decode(self.outputs, skip_special_tokens=True) # outputs[0] instead of output ?
        except AttributeError:
            raise Exception("AttributeError while trying to decode outputs skipping special tokens.")