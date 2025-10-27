from src.Framework.ModelInputPreparer.run import run


def main():
    # --- CONFIG ---
    logPartialResults = False
    run(logPartialResults)
    # --- end of config ---

if __name__ == '__main__':
    main()