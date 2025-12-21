try:
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.google.google import tokenizeAutoModelForGoogle0
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.qwen.qwen import tokenizeAutoModelForQwenAndSimilar0
    from src.Framework.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString_Qwen_Microsoft, getMessagesAsString_Google
    from src.Framework.HuggingFaceModelInferencer.modelname import MODEL_NAME
    from src.Framework.HuggingFaceModelInferencer.config import NUMBER_OF_DESIRED_ANSWERS, DEBUG_MODE
except Exception as e:
    from .orgs.google import tokenizeAutoModelForGoogle0
    from .orgs.qwen import tokenizeAutoModelForQwenAndSimilar0
    from MessagesAsASingleStringBuilder.Builder import getMessagesAsString_Qwen_Microsoft, getMessagesAsString_Google
    from modelname import MODEL_NAME
    from config import NUMBER_OF_DESIRED_ANSWERS, DEBUG_MODE

from transformers import AutoTokenizer


def _dbg(msg, *args, **kwargs):
    if DEBUG_MODE:
        print(msg, *args, **kwargs)

class TransformersApiHandler:
    def __init__(self):
        print('TransformersApiHandler() initalized')
        self.tokenizer = []
        self.model = []
        self.generatedIds = []
        self.response = []
        self.modelInputs = []

    def DoAutotokenizerFromPretrained(self):
        print("Executing AutoTokenizer.from_pretrained...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME)  # should be of type TextKwargs(), but TextKwargs is inaccessible from here for some reason
        print("AutoTokenizer.from_pretrained done!")



    def qwen(self, i, questions, creative: bool = False, max_new_tokens: int = 1):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print('====== QWEN PATH STARTED ======')

        # 1) Load model
        print(f"Loading model: {MODEL_NAME}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": "cpu"},
            dtype="bfloat16" # float32 or bfloat16 if supported
        )
        print("Model loaded:")
        print("  model class:", type(self.model))
        _dbg("  device map:", self.model.hf_device_map)

        # 2) Load tokenizer
        _dbg(f"Loading tokenizer: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _dbg("Tokenizer loaded:", self.tokenizer)

        # 3) Build prompt
        msgs = getMessagesAsString_Qwen_Microsoft( questions, i, NUMBER_OF_DESIRED_ANSWERS)
        print("MessagesAsString:", msgs)

        prompt = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        print("Prompt:")
        print(prompt)

        # 4) Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        _dbg("Tokenized inputs:")

        for k, v in inputs.items():
            _dbg(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
        _dbg("inputs object:", inputs)

        gen_kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": self.tokenizer.eos_token_id}
        gen_kwargs.update(_generate_args(creative))

        # 5) Generate
        print("Calling model.generate...")
        output = self.model.generate(
            **inputs,
            max_new_tokens=1
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
        decoded: list[str] = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        print("Decoded output:")
        for i, d in enumerate(decoded):
            print(f"  decoded[{i}]: {repr(d)}")


        print("====== QWEN PATH END ======")

        return decoded, generated_ids

    def google(self, i, questions, creative: bool = False, max_new_tokens: int = 1, use_torch_compile: bool = False):
        """
        Run google/gemma-2-2b-it (instruction-tuned).
        Returns (decoded_list, generated_ids_tensor)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import os

        _dbg("=== GEMMA PATH STARTED ===")
        _dbg("MODEL_NAME:", MODEL_NAME)

        # Load tokenizer (instruction-tuned -it -> use chat template)
        _dbg("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _dbg("Tokenizer:", self.tokenizer)

        # Optionally speed-up with torch.compile (docs recommend warm-up steps)
        # Using device_map + dtype or revision float16 is recommended when GPU available.
        load_kwargs = {"device_map": "auto"}
        # If GPU present, prefer float16/revision float16 (user can change)
        try:
            if torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["device_map"] = {"": "cpu"}
        except Exception:
            pass

        _dbg("Loading model with kwargs:", load_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
        _dbg("Model loaded:", type(self.model), "device_map:", getattr(self.model, "hf_device_map", None))

        if use_torch_compile:
            try:
                _dbg("Compiling model with torch.compile() (Gemma can benefit). Running 2 warm-up steps recommended.")
                # compile may speed up inference on supported PyTorch versions


                self.model.eval() # On CPU, torch.compile() is rarely worth it and sometimes slower.

                if use_torch_compile and torch.cuda.is_available():
                    self.model = torch.compile(self.model)
            except Exception as e:
                _dbg("torch.compile() failed or unsupported:", e)

        # Build prompt using chat template for instruction models
        msgs = getMessagesAsString_Google(questions, i, NUMBER_OF_DESIRED_ANSWERS)
        # Add an assistant message if missing

        _dbg("MessagesAsString:", msgs)

        _dbg("Applying chat template...")

        prompt = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        print("Prompt:\n" + prompt)

        # tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        _dbg("Tokenized inputs shapes:", {k: v.shape for k, v in inputs.items()})



        gen_kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": self.tokenizer.eos_token_id}
        gen_kwargs.update(_generate_args(creative))

        _dbg("Calling model.generate with:", gen_kwargs)
        output = self.model.generate(**inputs, **gen_kwargs)

        _dbg("Generate output shape:", output.shape)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[:, input_len:]
        _dbg("Sliced generated_ids shape:", generated_ids.shape)

        decoded: list[str] = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        _dbg("Decoded:", decoded)
        _dbg("=== GEMMA PATH END ===")
        return decoded, generated_ids

    def microsoft(self, i, questions, creative: bool = False, max_new_tokens: int = 1):
        """
        Run microsoft/phi-4-mini-instruct.
        Returns (decoded_list, generated_ids_tensor)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        import os

        _dbg("=== PHI-4 MINI PATH STARTED ===")
        _dbg("MODEL_NAME:", MODEL_NAME)

        # Deterministic seed (optional) for reproducibility in phi examples
        try:
            torch.manual_seed(0)
        except Exception:
            pass

        _dbg("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _dbg("Tokenizer:", self.tokenizer)

        # Use device_map="auto" and dtype hints if GPU available (phi docs show this pattern)

        offloadFolder = os.path.expanduser("~\\hf_offload")
        os.makedirs(offloadFolder, exist_ok=True)
        load_kwargs = {}
        try:
            if torch.cuda.is_available():
                load_kwargs["device_map"]= "auto"
                load_kwargs["torch_dtype"]= torch.float16
                load_kwargs["offload_folder"] = offloadFolder

            else:
                load_kwargs["device_map"]= {"": "cpu"}

        except Exception:
            pass

        _dbg("Loading model with kwargs:", load_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
        _dbg("Model loaded:", type(self.model), "device_map:", getattr(self.model, "hf_device_map", None))

        # Build prompt — phi-mini-instruct is instruction-tuned, use same chat-template approach if available
        msgs = getMessagesAsString_Qwen_Microsoft(questions, i, NUMBER_OF_DESIRED_ANSWERS)
        _dbg("MessagesAsString:", msgs)
        # Some tokenizers/models don't have apply_chat_template; guard it
        try:
            prompt = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # fallback: join roles into a single prompt
            prompt = "\n".join([m.get("content", "") for m in msgs])
        _dbg("Prompt:\n", prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        _dbg("Tokenized inputs shapes:", {k: v.shape for k, v in inputs.items()})


        gen_kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": self.tokenizer.eos_token_id}
        gen_kwargs.update(_generate_args(creative))

        _dbg("Calling model.generate with:", gen_kwargs)
        output = self.model.generate(**inputs, **gen_kwargs)

        _dbg("Generate output shape:", output.shape)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[:, input_len:]
        _dbg("Sliced generated_ids shape:", generated_ids.shape)

        decoded: list[str] = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        _dbg("Decoded:", decoded)
        _dbg("=== PHI-4 MINI PATH END ===")
        return decoded, generated_ids

    def batchDecodeGenerateFinalAnswer(self, convertedTensors):
        if self.tokenizer is None:
            print('failed to give self.tokenizer a value using AutoTokenizer.from_pretrained(MODEL_NAME.strip()).')
        try:
            print(convertedTensors)
            self.response: list[str] = self.tokenizer.batch_decode(convertedTensors,
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


# generation args helper
def _generate_args(creative_flag: bool):
    if creative_flag:
        return {"do_sample": True, "temperature": 0.8, "top_p": 0.9}
    else:
        return {"do_sample": False}