OS = 'Linux'
BASE_PATH = 'C:\WiC_dataset'
if OS == 'Linux':
    BASE_PATH = r'/home.local/valaki/Letöltések/WiC_dataset'
elif OS == 'Windows':
    BASE_PATH = r'C:\WiC_dataset'
print(BASE_PATH)