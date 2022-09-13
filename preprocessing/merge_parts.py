import os
import json
from pathlib import Path
from tqdm import tqdm

objects_path = './data/PartNet/objects'
object_ids = os.listdir(objects_path)

object_id = object_ids[1]

def merge_objs(obj_names, part_name, objs_path):
    all_vertices = []
    all_faces = []
    for obj_name in obj_names:
        vertices = []
        faces = []
        with open(f'{objs_path}/{obj_name}.obj', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('v'):
                    vertices.append(line)
                elif line.startswith('f'):
                    faces.append(line)
                else:
                    print('WARNING: unknown entry ' + line)
        all_vertices += vertices
        all_faces += faces
    merged_path = f'{objs_path}/merged'
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    obj_path = f'{merged_path}/{part_name}.obj'
    with open(obj_path, 'w') as f:
        f.writelines(all_vertices + all_faces)
    return obj_path

# Merge all obj files into full object
def get_objs(parts):
    objs = []
    for part in parts:
        if 'objs' in part:
            objs += part['objs']
        if 'children' in part:
            objs += get_objs(part['children'])
    return objs

for object_id in tqdm(object_ids):
    object_path = f'{objects_path}/{object_id}'
    objs_path = f'{object_path}/objs'

    with open(f'{object_path}/result.json') as f:
        result = json.load(f)

    obj = result[0]
    obj_objs = get_objs(obj['children'])
    full_obj_path = merge_objs(obj_objs, obj['name'], objs_path)

    # Merge part obj files into full parts
    with open(f'{object_path}/result_labeled.json') as f:
        result_labeled = json.load(f)

    obj_labeled = result_labeled[0]
    parts = obj_labeled['labeled_parts']
    for part in parts:
        obj_path = merge_objs(part['objs'], part['name'], objs_path)
        part['obj_path'] = obj_path
    obj_labeled['obj_path'] = full_obj_path

    with open(f'{objects_path}/{object_id}/result_labeled.json', 'w') as f:
        json.dump([obj_labeled], f)