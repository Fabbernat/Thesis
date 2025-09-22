from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub.errors import HFValidationError

try:
    from Resurrection.CloudRunnerNotebooks.MessagesAsASingleStringBuilder.Builder import getMessagesAsString
except Exception as e:
    print("Warning: Could not import getMessagesAsString:", e)
    getMessagesAsString = lambda: ""  # fallback: empty prompt


class TransformersApiHandler:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.generated_ids = None
        self.answer = None

        from Resurrection.CloudRunnerNotebooks.Config.Config import MODEL_NAME
        try:
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")
        except AttributeError as e:
            print("AttributeError in self.model.generate.", str(e))

        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME.strip())
        except HFValidationError as e:
            print("Invalid Hugging Face model name:", e)
            return  # shut down gracefully
        except Exception as e:
            print("Unexpected error while loading model:", e)
            return


        prompt = tokenizer.apply_chat_template(getMessagesAsString(), tokenize=False, add_generation_prompt=True)

        self.inputs = tokenizer([prompt], return_tensors="pt").to(self.model.device)

    def generateIds1(self):
        self.generated_ids = self.model.generate(**self.inputs, max_new_tokens=512)
        return self.generated_ids

    def convertIds2(self):
        self.generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(self.inputs.input_ids, self.generated_ids)
        ]

        return self.generated_ids

    def generateFinalAnswer3(self):
        self.answer = self.tokenizer.batch_decode(self.generated_ids, skip_special_tokens=True)[0]
        print(self.generated_ids)
        return self.answer or self.generated_ids or self.tokenizer # need to test all three

    def decodeOutputsSkippingSpecialTokens(self):
        try:
            self.tokenizer.decode(self.generated_ids, skip_special_tokens=True)
        except AttributeError as ae:
            raise Exception("AttributeError while trying to decode outputs skipping special tokens.", ae)