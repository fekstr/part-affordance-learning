import os
import pickle
from pathlib import Path
from tqdm import tqdm

with open('./data/object_ids.pkl', 'rb') as f:
    obj_id_map = pickle.load(f)

data_path = './data/PartNet/data_v0'
selected_path = './data/PartNet/selected'

dirs = tuple(os.listdir(data_path))
for obj_name, ids in tqdm(obj_id_map.items()):
    dest_path = f'{selected_path}/{obj_name}'
    Path(dest_path).mkdir(parents=True, exist_ok=True) 
    for id in ids:
        if id in dirs:
            os.rename(f'{data_path}/{id}', f'{dest_path}/{id}')