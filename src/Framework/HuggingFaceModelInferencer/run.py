try:
    from src.Framework.HuggingFaceModelInferencer.ThirdPartyApiHandler import TorchApiHandler as TorchApiHandler
except Exception:
    from ThirdPartyApiHandler import TorchApiHandler as TorchApiHandler

from pathlib import Path

def run():
    print('run started')
    TorchApiHandler.TorchApiHandler().handleRequest()
