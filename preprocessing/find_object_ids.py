import os
import json
from collections import defaultdict
from tqdm import tqdm
import pickle

# Path to original extracted data
data_path = './data/PartNet/data_v0'

obj_ids = os.listdir(data_path)

def get_lines(path):
    with open(path) as f:
        lines = [line.replace('\n', '') for line in f.readlines()]
    return lines

def get_names(objs):
    names = []
    for obj in objs:
        names.append(obj['name'])
        if 'children' in obj:
            names += get_names(obj['children'])
    return names 

train_objects = get_lines('./data/train_objects.txt')
test_objects = get_lines('./data/test_objects.txt')
objects = set(train_objects + test_objects)

# Find IDs corresponding to selected objects
obj_id_map = defaultdict(lambda: list())
names = set()
texts = set()
top_level_names = set()
for obj_id in tqdm(obj_ids):
    obj_path = f'{data_path}/{obj_id}'
    with open(f'{obj_path}/result.json') as f:
        result = json.load(f)
    obj_names = get_names(result)
    top_level_names.add(result[0]['name'])
    texts.add(result[0]['text'])
    names.update(obj_names)
    for obj in objects:
        if obj in obj_names:
            obj_id_map[obj].append(obj_id)

# Find ids appearing in multiple classes
id_classes = defaultdict(lambda: list())
for class_name, class_ids in obj_id_map.items():
    for class_id in class_ids:
        id_classes[class_id].append(class_name)

# Intersections
intersections = [tuple(combo) for combo in id_classes.values() if len(combo) > 1]
counts = defaultdict(lambda: 0)
for combo in intersections:
    counts[combo] += 1
counts

# Delete objects with multiple unrelated matches
combo_id_map = defaultdict(lambda: list())
for id, combo in id_classes.items():
    combo_id_map[tuple(combo)].append(id)

ids_to_delete = combo_id_map[('table', 'bed')] + combo_id_map[('chair', 'bed')]
obj_id_map['table'] = [id for id in obj_id_map['table'] if id not in ids_to_delete]
obj_id_map['bed'] = [id for id in obj_id_map['bed'] if id not in ids_to_delete]
obj_id_map['chair'] = [id for id in obj_id_map['chair'] if id not in ids_to_delete]

# Ensure each object is only in one class 
desk_ids = combo_id_map[('table', 'desk')]
backpack_ids = combo_id_map[('bag', 'backpack')]

obj_id_map['table'] = [id for id in obj_id_map['table'] if id not in desk_ids]
obj_id_map['bag'] = [id for id in obj_id_map['bag'] if id not in backpack_ids]

# Summary
summary = { obj: len(ids) for obj, ids in obj_id_map.items() }
summary
sum([i for i in summary.values()])

# Save the extracted IDs
with open('./data/object_ids.pkl', 'wb') as f:
    pickle.dump(dict(obj_id_map), f)

# Load as follows
# with open('./data/object_ids.pkl', 'rb') as f:
#     obj_id_map = pickle.load(f)

