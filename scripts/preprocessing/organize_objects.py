import os
import pickle
from pathlib import Path
import random
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


def move_ids(src, dest, ids):
    Path(dest).mkdir(parents=True, exist_ok=True)
    for id in tqdm(ids):
        if id in tuple(os.listdir(src)):
            os.rename(os.path.join(src, id), os.path.join(dest, id))


move_ids(selected_path, os.path.join(selected_path, 'train'), train_ids)
move_ids(selected_path, os.path.join(selected_path, 'test'), test_ids)

# Create stratified validation set
train_obj_id_map = {
    obj: ids
    for obj, ids in obj_id_map.items() if obj in train_objects
}

valid_ids = []
for ids in train_obj_id_map.values():
    valid_ids += random.sample(ids, int(0.1 * len(ids)))

move_ids(os.path.join(selected_path, 'train'),
         os.path.join(selected_path, 'valid'), valid_ids)
