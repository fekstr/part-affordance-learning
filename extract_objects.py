import pickle
import os
from tqdm import tqdm
from zipfile import ZipFile

data_path = './data/PartNet/data_v0.zip'
archive = ZipFile(data_path)

with open('./data/object_ids.pkl', 'rb') as f:
    obj_id_map = pickle.load(f)

all_ids = []
for ids in obj_id_map.values():
    all_ids += ids
allowed_prefixes = tuple([f'data_v0/{id}/' for id in all_ids])

for name in tqdm(archive.namelist()):
    if name.startswith(allowed_prefixes):
        archive.extract(name, './data/PartNet')

os.rename('./data/PartNet/data_v0', './data/PartNet/objects')
