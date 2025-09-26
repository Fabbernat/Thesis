# Optional global main running all three submodules, if you don't want to click three times (not recommended)
from Resurrection import ModelInputPreparer, HuggingFaceModelInferencer, ModelOutputProcessor


def main():
    try:
        ModelInputPreparer.main.main()
        HuggingFaceModelInferencer.main.main()
        ModelOutputProcessor.main.main()
    except AttributeError as ae:
        print('This is not working. Run the modules one by one instead.')
if __name__ == "__main__":
    main()
