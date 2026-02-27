

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
        print(f'Sorry, this is not working at moment, but the devs know about this issue. Write a mail to job.fabbernat@gmail.com to make them hurry! \n Exception occurred: {e}.\nRun the three modules one by one instead.')
if __name__ == "__main__":
    main()
