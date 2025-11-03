from Framework import ModelInputPreparer, HuggingFaceModelInferencer, ModelOutputProcessor


# BETA: Optional global main running all three submodules if you don't want to click three times (not recommended, since the second module often runs to errors or bugs)
def main():
    try:
        ModelInputPreparer.main.main()
        HuggingFaceModelInferencer.main.main()
        ModelOutputProcessor.main.main()
    except AttributeError as ae:
        print('This is not working. Run the three modules one by one instead. Error:', ae)

if __name__ == "__main__":
    main()
