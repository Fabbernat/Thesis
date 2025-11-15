try:
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.google.google import tokenizeAutoModelForGoogle0
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler.qwen.qwen import tokenizeAutoModelForQwenAndSimilar0, \
        generateIds1, convertIds2
    from src.Framework.HuggingFaceModelInferencer.MessagesAsASingleStringBuilder.Builder import getMessagesAsString
    from src.Framework.HuggingFaceModelInferencer.modelname import MODEL_NAME
except Exception as e:
    from .google.google import tokenizeAutoModelForGoogle0
    from .qwen.qwen import  tokenizeAutoModelForQwenAndSimilar0
    from MessagesAsASingleStringBuilder.Builder import getMessagesAsString
    from modelname import MODEL_NAME

from transformers import AutoTokenizer

class TransformersApiHandler:
    def __init__(self):
        print('TransformersApiHandler() initalized')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # should be of type TextKwargs(), but TextKwargs is inaccessible from here for some reason
        self.model = object
        self.generatedIds = object
        self.response = object
        self.modelInputs = object

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
        '''
        C:\PycharmProjects\Thesis\.venv1\Scripts\python.exe C:\PycharmProjects\Thesis\src\Framework\HuggingFaceModelInferencer\main.py
Answer all 15 questions with either `Yes` or `No`.

Is cuda available?  False
 ** RUNTIME ENVIRONMENT INFO **
sys.path:  ['C:\\PycharmProjects\\Thesis\\src\\Framework\\HuggingFaceModelInferencer', 'C:\\Program Files\\JetBrains\\PyCharm 2025.2\\plugins\\python-ce\\helpers\\pycharm_display', 'C:\\Python312\\python312.zip', 'C:\\Python312\\DLLs', 'C:\\Python312\\Lib', 'C:\\Python312', 'C:\\PycharmProjects\\Thesis\\.venv1', 'C:\\PycharmProjects\\Thesis\\.venv1\\Lib\\site-packages', 'C:\\Program Files\\JetBrains\\PyCharm 2025.2\\plugins\\python-ce\\helpers\\pycharm_matplotlib_backend', 'C:\\Program Files\\JetBrains\\PyCharm 2025.2\\plugins\\python-ce\\helpers\\pycharm_altair_backend', 'C:\\Program Files\\JetBrains\\PyCharm 2025.2\\plugins\\python-ce\\helpers\\pycharm_plotly_backend']
 ** end of runtime environment info **
main started
run started
TorchApiHandler initialized
TorchApiHandler.handleRequest() started
TransformersApiHandler() initalized
qwen path chosen
Some parameters are on the meta device because they were offloaded to the disk and cpu.
tokenizer before=Qwen2TokenizerFast(name_or_path='qwen/qwen2.5-0.5b-instruct', vocab_size=151643, model_max_length=131072, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151646: AddedToken("<|object_ref_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151647: AddedToken("<|object_ref_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151648: AddedToken("<|box_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151649: AddedToken("<|box_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151650: AddedToken("<|quad_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151651: AddedToken("<|quad_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151652: AddedToken("<|vision_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151653: AddedToken("<|vision_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151654: AddedToken("<|vision_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151655: AddedToken("<|image_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151656: AddedToken("<|video_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151657: AddedToken("<tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151658: AddedToken("</tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151659: AddedToken("<|fim_prefix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151660: AddedToken("<|fim_middle|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151661: AddedToken("<|fim_suffix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151662: AddedToken("<|fim_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151663: AddedToken("<|repo_name|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151664: AddedToken("<|file_sep|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
}
)
tokenizer after=Qwen2TokenizerFast(name_or_path='qwen/qwen2.5-0.5b-instruct', vocab_size=151643, model_max_length=131072, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|im_start|>', '<|im_end|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151646: AddedToken("<|object_ref_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151647: AddedToken("<|object_ref_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151648: AddedToken("<|box_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151649: AddedToken("<|box_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151650: AddedToken("<|quad_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151651: AddedToken("<|quad_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151652: AddedToken("<|vision_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151653: AddedToken("<|vision_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151654: AddedToken("<|vision_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151655: AddedToken("<|image_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151656: AddedToken("<|video_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	151657: AddedToken("<tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151658: AddedToken("</tool_call>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151659: AddedToken("<|fim_prefix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151660: AddedToken("<|fim_middle|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151661: AddedToken("<|fim_suffix|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151662: AddedToken("<|fim_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151663: AddedToken("<|repo_name|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
	151664: AddedToken("<|file_sep|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=False),
}
)
Questions file path=C:\PycharmProjects\Thesis\src\Framework\HuggingFaceModelInferencer\data/questions.in
reading: Does the word "joke" mean the same thing in sentences "I regarded his campaign for mayor as a joke." and "He told a very funny joke."?
Does the word "joke" mean the same thing in sentences "He told a very funny joke." and "I regarded his campaign for mayor as a joke."?

 *** The prompt: ***
Answer all 15 questions with either `Yes` or `No`.

---------------
Does the word "joke" mean the same thing in sentences "I regarded his campaign for mayor as a joke." and "He told a very funny joke."?
Does the word "joke" mean the same thing in sentences "He told a very funny joke." and "I regarded his campaign for mayor as a joke."?
 *** End of the prompt ***

Writing prompt to: C:\PycharmProjects\Thesis\src\Framework\HuggingFaceModelInferencer\data\prompt.out
Exception in handleModelSpecificActions: name 'generateIds1' is not defined
[]
Traceback (most recent call last):
  File "C:\PycharmProjects\Thesis\src\Framework\HuggingFaceModelInferencer\main.py", line 31, in <module>
    main()
  File "C:\PycharmProjects\Thesis\src\Framework\HuggingFaceModelInferencer\main.py", line 22, in main
    run.run()
  File "C:\PycharmProjects\Thesis\src\Framework\HuggingFaceModelInferencer\run.py", line 7, in run
    TorchApiHandler.TorchApiHandler().handleRequest()
  File "C:\PycharmProjects\Thesis\src\Framework\HuggingFaceModelInferencer\ThirdPartyApiHandler\TorchApiHandler.py", line 31, in handleRequest
    modelResponses, generated_ids, tokenizer = self.transformersApiHandler.batchDecodeGenerateFinalAnswer3(self.convertedIdsTensors)
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\PycharmProjects\Thesis\src\Framework\HuggingFaceModelInferencer\ThirdPartyApiHandler\TransformersApiHandler.py", line 46, in batchDecodeGenerateFinalAnswer3
    self.response = self.tokenizer.batch_decode(convertedTensors, skip_special_tokens=True)[
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IndexError: list index out of range

Process finished with exit code 1
        :param convertedTensors:
        :return:
        '''
        if self.tokenizer is None:
            print('failed to give self.tokenizer a value using AutoTokenizer.from_pretrained(MODEL_NAME.strip()).')
        try:
            print(convertedTensors)
            self.response = self.tokenizer.batch_decode(convertedTensors, skip_special_tokens=True)[
                0]  # [NUMBER_OF_DESIRED_ANSWERS] # [0] makes the answers longer for some reason, so [NUMBER_OF_DESIRED_ANSWERS] is not needed.
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