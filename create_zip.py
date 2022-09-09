from zipfile import ZipFile, ZIP_DEFLATED
import os
import pickle

with ZipFile('selected.zip', 'w') as zip_obj:
    data_path = './data/PartNet/data_v0'
    for obj in ['1000', '1095']:
        for folder, subfolders, filenames in os.walk(f'{data_path}/{obj}'):
            for filename in filenames:
                filepath = os.path.join(folder, filename)
                zip_obj.write(filepath, filepath.replace(data_path, ''), ZIP_DEFLATED)