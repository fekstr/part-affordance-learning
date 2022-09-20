import os
import pickle
from pathlib import Path
from tqdm import tqdm
from preprocessing.utils import load_split

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

# Organize in train and test folders
selected_path = './data/PartNet/selected_objects'

train_objects, test_objects, _ = load_split()
train_ids = []
test_ids = []
for obj, ids in obj_id_map.items():
    if obj in train_objects:
        train_ids += ids
    elif obj in test_objects:
        test_ids += ids
    else:
        raise Exception('Unknown object', obj)

selected_dirs = tuple(os.listdir(selected_path))

def move_ids(src, dest, ids):
    Path(dest).mkdir(parents=True, exist_ok=True) 
    for id in tqdm(ids):
        if id in selected_dirs:
            os.rename(
                os.path.join(selected_path, id),
                os.path.join(dest, id)
            )


move_ids(selected_path, os.path.join(selected_path, 'train'), train_ids)
move_ids(selected_path, os.path.join(selected_path, 'test'), test_ids)
