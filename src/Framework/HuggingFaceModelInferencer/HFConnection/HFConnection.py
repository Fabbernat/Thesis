import requests

class HFConnectionHandler:
    def __init__(self, base_url: str = "https://huggingface.co/"):
        self.base_url = base_url

    def connect(self):
        try:
            response = requests.get(self.base_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Successfully connected to {self.base_url}")
            else:
                print(f"⚠️ Connection failed to {self.base_url} "
                      f"(status code: {response.status_code})")
        except requests.RequestException as e:
            print(f"❌ Connection failed to {self.base_url}: {e}")

    def accessModel(self, MODEL_NAME: str):
        pass

hfcHandler = HFConnectionHandler()
hfcHandler.connect()