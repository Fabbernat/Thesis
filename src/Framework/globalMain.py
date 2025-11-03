CONNECTED = False
try:
    from src.Framework.ModelInputPreparer import main as m1
    from src.Framework.HuggingFaceModelInferencer import main as m2
    from src.Framework.ModelOutputProcessor import main as m3
except ModuleNotFoundError as mne:
    print('ModuleNotFoundError', mne)
    from .ModelInputPreparer import main as m1
    from .HuggingFaceModelInferencer import main as m2
    from .ModelOutputProcessor import main as m3

# BETA: Optional global main running all three submodules if you don't want to click three times (not recommended, since the second module often runs to errors or bugs)
def main():
    CONNECTED = True
    try:
        m1.main()
        m2.main()
        m3.main()
    except AttributeError as ae:
        print('This is not working. Run the three modules one by one instead. Error:', ae)

if __name__ == "__main__":
    main()
