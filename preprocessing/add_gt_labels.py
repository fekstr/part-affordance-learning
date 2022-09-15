import os
import pandas as pd
from collections import defaultdict
import json
from tqdm import tqdm

from preprocessing.utils import get_obj_name

gt = pd.read_pickle('./data/gt.pkl')
gt = gt.drop(columns='body_part')
gt = gt.drop_duplicates().reset_index(drop=True)

# Create a map from (object, object_part) to affordance[]
object_part_affordance_map = defaultdict(lambda: defaultdict(lambda: list()))
for row in gt.iterrows():
    entry = row[1]
    object_part_affordance_map[entry['object']][entry['object_part']].append(entry['affordance'])

for key in object_part_affordance_map.keys():
    object_part_affordance_map[key] = dict(object_part_affordance_map[key])
object_part_affordance_map = dict(object_part_affordance_map)

# Iterate through part hierarchy adding labels to matching parts
objects_path = './data/PartNet/selected_objects'
object_ids = os.listdir(objects_path)

def label_children(children, part_affordance_map):
    for child in children:
        text = child['text'].lower()
        if text in part_affordance_map.keys():
            child['affordances'] = part_affordance_map[text]
        if 'children' in child:
            label_children(child['children'], part_affordance_map)

def lift_labeled_parts(parts):
    labeled_parts = []
    for part in parts:
        if 'affordances' in part:
            labeled_parts.append(part)
        if 'children' in part:
            labeled_parts += lift_labeled_parts(part['children'])
    return labeled_parts

def get_objs(part):
    objs = []
    if 'objs' in part:
        objs += part['objs']
    if 'children' in part:
        for subpart in part['children']:
            objs += get_objs(subpart)
    return objs

for object_id in tqdm(object_ids):
    with open(f'{objects_path}/{object_id}/result.json') as f:
        result = json.load(f)
    obj = result[0]
    obj_name = get_obj_name(obj, object_part_affordance_map.keys())
    part_affordance_map = object_part_affordance_map[obj_name]
    label_children(obj['children'], part_affordance_map)
    obj['labeled_parts'] = lift_labeled_parts(obj['children'])
    for part in obj['labeled_parts']:
        objs = get_objs(part)
        part['objs'] = objs
        part.pop('children', None)
    with open(f'{objects_path}/{object_id}/result_labeled.json', 'w') as f:
        json.dump([obj], f)