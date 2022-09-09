import os
import json
from collections import defaultdict
from tqdm import tqdm
import pickle
import shutil
from pathlib import Path

# Path to mounted remote
base_path = './PartNet/data_v0'

obj_ids = os.listdir('./PartNet/data_v0')

def get_lines(path):
    with open(path) as f:
        lines = [line.replace('\n', '') for line in f.readlines()]
    return lines

train_objects = get_lines('./data/train_objects.txt')
test_objects = get_lines('./data/test_objects.txt')
objects = set(train_objects + test_objects)

# Find IDs corresponding to selected objects
obj_id_map = defaultdict(lambda: list())
for obj_id in tqdm(obj_ids):
    obj_path = f'{base_path}/{obj_id}'
    with open(f'{obj_path}/result.json') as f:
        result = json.load(f)
    obj_name = result[0]['name']
    if obj_name in objects:
        obj_id_map[obj_name].append(obj_id)

# Save the extracted IDs
with open('./data/object_ids.pkl', 'wb') as f:
    pickle.dump(dict(obj_id_map), f)

with open('./data/object_ids.pkl', 'rb') as f:
    obj_id_map = pickle.load(f)


# TODO:
# [ ] Copy all relevant objects using these IDs
# [ ] Search for label matches in their part hierarchies