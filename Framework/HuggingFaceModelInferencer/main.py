import sys; print(sys.path)
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from Framework.HuggingFaceModelInferencer.run import run


def main():
    print('main started')
    run()
'''
All models require pip install torch transformers accelerate
Some models require  pip install torch transformers accelerate hf_xetm optimum
'''
if __name__ == '__main__':
    main()