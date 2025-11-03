CONNECTED = False
try:
    from src.Framework.ModelInputPreparer.main import main as m1
    from src.Framework.HuggingFaceModelInferencer.main import main as m2
    from src.Framework.ModelOutputProcessor.main import main as m3
except ModuleNotFoundError as mne:
    print('ModuleNotFoundError', mne)

try:
    from .ModelInputPreparer.main import main as m1
    from .HuggingFaceModelInferencer.main import main as m2
    from .ModelOutputProcessor.main import main as m3
except ImportError as ie:
    print('ImportError', ie)

# BETA: Optional global main running all three submodules if you don't want to click three times (not recommended, since the second module often runs to errors or bugs)
def main():
    CONNECTED = True
    try:
        m1()
        m2()
        m3()
    except AttributeError as ae:
        print('This is not working. Run the three modules one by one instead. Error:', ae)
    except NameError as ne:
        print('NameError', ne)

if __name__ == "__main__":
    main()
