import os
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def merge_objs(obj_names, part_name, objs_path, path_only=False):
    merged_path = f'{objs_path}/merged_crude'
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    obj_path = f'{merged_path}/{part_name}.obj'

    if path_only:
        return obj_path

    all_vertices = []
    all_faces = []
    offset = 0
    for obj_name in obj_names:
        vertices = []
        faces = []
        with open(f'{objs_path}/{obj_name}.obj', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('v'):
                    vertices.append(line)
                elif line.startswith('f'):
                    line_w_offset = ['f']
                    for token in line.split()[1:]:
                        line_w_offset.append(str(int(token) + offset))
                    line_w_offset = ' '.join(line_w_offset)
                    line_w_offset += '\n'
                    faces.append(line_w_offset)
                else:
                    print('WARNING: unknown entry ' + line)
        all_vertices += vertices
        all_faces += faces
        offset += len(vertices)
    with open(obj_path, 'w') as f:
        f.writelines(all_vertices + all_faces)
    return obj_path


def get_objs(results_after_merging):
    part = results_after_merging[0]
    objs = []
    objs.append({'name': 'full', 'objs': part['objs']})
    while 'children' in part and len(part['children']) == 1:
        part = part['children'][0]
    if 'children' in part:
        for child in part['children']:
            objs.append({'name': child['name'], 'objs': child['objs']})
    return objs


def get_obj_name(result):
    obj = result[0]
    if 'children' in obj and len(obj['children']) == 1:
        obj = obj['children'][0]
    return obj['name']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src')
    parser.add_argument('--id')
    parser.add_argument('--path-only',
                        action='store_true',
                        help='Update paths without merging objs')
    args = parser.parse_args()

    objects_path = args.src
    object_ids = [args.id] if args.id else os.listdir(objects_path)
    for object_id in tqdm(object_ids):
        object_path = f'{objects_path}/{object_id}'
        objs_path = f'{object_path}/objs'

        with open(f'{object_path}/result_after_merging.json') as f:
            result_after_merging = json.load(f)

        # Add the top level object for the full mesh, then descend to the first level where there are multiple parts
        parts = get_objs(result_after_merging)
        obj_name = get_obj_name(result_after_merging)

        merged_parts = []
        name_counts = defaultdict(lambda: 0)
        for part in parts:
            part_obj_path = merge_objs(
                part['objs'],
                part['name'] + '_' + str(name_counts[part['name']]), objs_path,
                args.path_only)
            if part['name'] == 'full':
                full_obj_path = part_obj_path
            else:
                merged_parts.append({
                    'name': part['name'],
                    'obj_path': part_obj_path
                })
                name_counts[part['name']] += 1

        result = {
            'name': obj_name,
            'obj_path': full_obj_path,  # Path to full obj
            'parts': merged_parts
        }

        with open(os.path.join(objects_path, object_id, 'result_merged.json'),
                  'w') as f:
            json.dump(result, f)
