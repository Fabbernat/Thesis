# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Always adjust the path to match your operating system!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

OS = 'Windows'

# Path to the local root of https://pilehvar.github.io/wic/package/WiC_dataset.zip
BASE_PATH = 'C:\WiC_dataset'

if OS == 'Linux':
    RECOMMENDED_PATH = r'/home.local/valaki/Letöltések/WiC_dataset'
elif OS == 'Windows':
    RECOMMENDED_PATH = r'C:\WiC_dataset'

print(f'The app will search the dataset at{BASE_PATH}')

def apply_recommended_path():
    BASE_PATH = RECOMMENDED_PATH

def set_custom_path(PATH):
    BASE_PATH = PATH