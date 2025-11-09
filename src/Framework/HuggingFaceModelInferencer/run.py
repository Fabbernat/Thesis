from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler import TorchApiHandler as TorchApiHandler

def run():
    print('run started')
    TorchApiHandler.TorchApiHandler().handleRequest()
