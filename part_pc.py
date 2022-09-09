# Input: object directory
# Output: [(part_point_cloud, part_label)]
import pandas as pd
from pyntcloud import PyntCloud
import json
from pathlib import Path
from collections import defaultdict

data_path = './data/PartNet/data_v0'

def get_objs(part):
    objs = []
    if 'objs' in part:
        objs += part['objs']
    if 'children' in part:
        for child in part['children']:
            objs += get_objs(child)
    return objs

def merge_objs(object_id: int, gt: pd.DataFrame):
    object_path = f'{data_path}/{str(object_id)}'
    with open(f'{object_path}/result.json') as f:
        result = json.load(f)

    main_object = result[0]
    parts = main_object['children']

    # Extract parts for which affordance labels are available
    main_object['name']
    obj_gt = gt[gt['object'] == main_object['name']]
    part_affordance_map = defaultdict(lambda: set())
    for row in obj_gt.iterrows():
        part = row[1]['object_part']
        part_affordance_map[part].add(row[1]['affordance'])
    part_affordance_map

    # Map top-level components to corresponding obj files
    parts_with_objs = []
    for part in parts:
        objs = get_objs(part)
        part['objs'] = objs
        part.pop('children', None)
        parts_with_objs.append(part)

    # Merge obj files by part
    Path(f'{object_path}/objs/merged').mkdir(parents=True, exist_ok=True)
    obj_vertices = []
    obj_faces = []
    for part in parts_with_objs:
        vertices = []
        faces = []
        for obj_name in part['objs']:
            with open(f'{object_path}/objs/{obj_name}.obj', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('v'):
                        vertices.append(line)
                    elif line.startswith('f'):
                        faces.append(line)
                    else:
                        print('WARNING: unknown entry ' + line)
        obj_path = f'{object_path}/objs/merged/{part["name"]}.obj'
        with open(obj_path, 'w') as f:
            f.writelines(vertices + faces)
        part['obj_path'] = obj_path
        part.pop('objs', None)
        obj_vertices += vertices
        obj_faces += faces

    main_obj_path = f'{object_path}/objs/merged/{main_object["name"]}.obj'
    with open(main_obj_path, 'w') as f:
        f.writelines(obj_vertices + obj_faces)

    main_object['parts'] = parts_with_objs
    main_object['obj_path'] = main_obj_path
    main_object.pop('children', None)

    return main_object

# Create object-part-affordance map
gt = pd.read_pickle('./data/gt.pkl')
gt = gt.set_index(['object', 'object_part'])
gt = gt.drop(columns='body_part')
gt = gt.drop_duplicates()

with open('./data/affordances.txt') as f:
    lines = f.readlines()
    affordances = [affordance.replace('\n', '') for affordance in lines]

gt = gt[gt['affordance'].isin(affordances)]


    
obj = merge_objs(1095)

 


# TODO:
# [ ] Merge obj files for the same top-level part
# [ ] Sample n points from each part
# [ ] Save (part_point_cloud, part_label) pairs