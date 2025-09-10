from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.errors import HFValidationError

from Resurrection.CloudRunnerNotebooks.ThesisCloudRunner import messages
from Resurrection.CloudRunnerNotebooks.run import MODEL_NAME


class Transformers3rdPartyApiHandler(object):
    def __init__(self):
        self.tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME.strip())
        except HFValidationError as e:
            print("Invalid Hugging Face model name:", e)
            return  # shut down gracefully
        except Exception as e:
            print("Unexpected error while loading model:", e)
            return

        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

        prompt = tokenizer.apply_chat_template(messages, tokenize=False)

        self.inputs = tokenizer(prompt, return_tensors="pt")

    def generateAnswersFor(self, messages):
        self.model.generate(**self.inputs, max_new_tokens=50)

    def decodeOutputsSkippingSpecialTokens(self):
        self.tokenizer.decode(outputs[0], skip_special_tokens=True)