# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Always adjust the path to match your operating system!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import os

OS = 'Windows'

# Path to the local root of https://pilehvar.github.io/wic/package/WiC_dataset.zip
# TODO change to C:\WiC_dataset\ and refactor all references
BASE_DIR = r'C:\WiC_dataset'
os.makedirs(BASE_DIR, exist_ok=True)

if OS == 'Linux':
    RECOMMENDED_PATH = r'/home.local/valaki/Letöltések/WiC_dataset'
elif OS == 'Windows':
    RECOMMENDED_PATH = r'C:\WiC_dataset'

print(f'The app will search the dataset at `{BASE_DIR}`. You can set a different path in PATH.py')

def apply_recommended_path():
    BASE_PATH = RECOMMENDED_PATH

def set_custom_path(PATH):
    BASE_PATH = PATH