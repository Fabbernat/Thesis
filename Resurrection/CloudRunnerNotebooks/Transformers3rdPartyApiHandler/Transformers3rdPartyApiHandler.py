import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.errors import HFValidationError

from Resurrection.CloudRunnerNotebooks.MessagesAsASingleStringMaker.MessagesAsASingleStringMaker import \
    getMessagesAsString


class Transformers3rdPartyApiHandler(object):
    def __init__(self):
        self.tokenizer = None
        self.model = None

        from Resurrection.CloudRunnerNotebooks.run import MODEL_NAME
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


        prompt = tokenizer.apply_chat_template(getMessagesAsString(), tokenize=False) # messages instead of None

        self.inputs = tokenizer(prompt, return_tensors="pt")

    def generateAnswers(self):
        self.model.generate(**self.inputs, max_new_tokens=50)

    def decodeOutputsSkippingSpecialTokens(self):
        self.tokenizer.decode(None, skip_special_tokens=True) # outputs[0] instead of None