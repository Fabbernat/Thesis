

# BETA: Optional global main running all three submodules if you don't want to click three times (not recommended, since the second module often runs to errors or bugs)
def main():
    try:
        from src.Framework import ModelInputPreparer, HuggingFaceModelInferencer, ModelOutputProcessor
        ModelInputPreparer.main.main(True)
        HuggingFaceModelInferencer.main.main(True)
        ModelOutputProcessor.main.main(True)
    except AttributeError as ae:
        print(f'AttributeError: {ae}')
    except Exception as e:
        print(f'An exception occurred: {str(e)}.\nRun the three modules one by one instead.')
if __name__ == "__main__":
    main()
