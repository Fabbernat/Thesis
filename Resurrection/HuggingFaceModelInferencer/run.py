from Resurrection.HuggingFaceModelInferencer.ThirdPartyApiHandler.TorchApiHandler import TorchApiHandler


def run():
    print('run started')
    TorchApiHandler().handleRequest()
