import os
import pickle
from pathlib import Path
from tqdm import tqdm

with open('./data/object_ids.pkl', 'rb') as f:
    obj_id_map = pickle.load(f)

all_ids = []
for obj_ids in obj_id_map.values():
    all_ids += obj_ids

data_path = './data/PartNet/data_v0'
selected_path = './data/PartNet/selected_objects'

dirs = tuple(os.listdir(data_path))
Path(selected_path).mkdir(parents=True, exist_ok=True) 
for id in tqdm(all_ids):
    if id in dirs:
        os.rename(f'{data_path}/{id}', f'{selected_path}/{id}')

os.rename(data_path, './data/PartNet/other_objects')