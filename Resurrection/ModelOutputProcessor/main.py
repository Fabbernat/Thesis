from Resurrection.ModelOutputProcessor.TernaryClassifier import TernaryClassifier

TESTFILE_LENGTH = 2800


def main():
    tf = TernaryClassifier()
    tf.classify()


if __name__ == '__main__':
    main()
