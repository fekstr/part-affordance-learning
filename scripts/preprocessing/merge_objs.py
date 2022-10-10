import os
import json
from pathlib import Path
from tqdm import tqdm


def merge_objs(obj_names, part_name, objs_path, path_only=False):
    merged_path = f'{objs_path}/merged_crude'
    Path(merged_path).mkdir(parents=True, exist_ok=True)
    obj_path = f'{merged_path}/{part_name}.obj'

    if path_only:
        return obj_path

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

        with open(f'{object_path}/result.json') as f:
            result = json.load(f)

        obj = result[0]
        obj_objs = get_objs(obj['children'])
        full_obj_path = merge_objs(obj_objs, 'full', objs_path, args.path_only)

        obj_merged = result[0]
        children = obj_merged['children']
        if len(children) == 1 and 'children' in children[0]:
            obj_merged = children[0]
            children = children[0]['children']
        obj_merged['obj_path'] = full_obj_path
        for child in children:
            child_objs = get_objs([child])
            child_obj_path = merge_objs(child_objs, child['name'], objs_path,
                                        args.path_only)
            child['obj_path'] = child_obj_path

        with open(os.path.join(objects_path, object_id, 'result_merged.json'),
                  'w') as f:
            json.dump([obj_merged], f)
