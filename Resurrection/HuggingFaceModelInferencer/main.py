import sys; print(sys.path)
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from Resurrection.HuggingFaceModelInferencer.run import run


def main():
    run()
'''
requires pip install torch, transformers, accelerate
'''
if __name__ == '__main__':
    main()