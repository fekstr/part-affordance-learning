import os
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import itertools
import shutil


def merge_objs(obj_names, part_name, objs_path, dest, path_only=False):
    merged_path = f'{dest}/objs'
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


def find_object(part, names):
    if part['name'] in names:
        # Found matching part
        return part
    elif 'children' not in part:
        # The object does not exist in the part label map
        return
    else:
        # Continue searching in children
        for child in part['children']:
            match = find_object(child, names)
            if match is not None:
                return match


def find_objs(part, names, objs):
    if part['name'] in names:
        objs.append({'name': part['name'], 'objs': part['objs']})
    elif 'children' not in part:
        return
    else:
        for child in part['children']:
            find_objs(child, names, objs)


def get_objs(results_after_merging, part_labels):
    part = find_object(results_after_merging[0], set(part_labels.keys()))
    if part is None:
        return None, None
    objs = []
    find_objs(part, set(part_labels[part['name']].keys()), objs)
    labeled_objs = itertools.chain(*[obj['objs'] for obj in objs])
    unlabeled_objs = set(part['objs']).difference(labeled_objs)
    if len(unlabeled_objs) > 0:
        objs.append({'name': 'other', 'objs': unlabeled_objs})
    objs.append({'name': 'full', 'objs': part['objs']})
    return objs, part['name']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src')
    parser.add_argument('--dest')
    parser.add_argument('--part-labels-path')
    parser.add_argument('--object-class')
    parser.add_argument('--id')
    parser.add_argument('--path-only',
                        action='store_true',
                        help='Update paths without merging objs')
    args = parser.parse_args()

    with open(args.part_labels_path) as f:
        part_labels = json.load(f)

    if args.object_class:
        part_labels = {
            c: v
            for c, v in part_labels.items() if c == args.object_class
        }

    objects_path = args.src
    object_ids = [args.id] if args.id else os.listdir(objects_path)
    for object_id in tqdm(object_ids):
        object_path = f'{objects_path}/{object_id}'
        objs_path = f'{object_path}/objs'

        with open(f'{object_path}/result_after_merging.json') as f:
            result_after_merging = json.load(f)

        parts, obj_name = get_objs(result_after_merging, part_labels)
        if parts is None:
            continue

        shutil.rmtree(os.path.join(args.dest, object_id))

        merged_parts = []
        name_counts = defaultdict(lambda: 0)
        for part in parts:
            part_obj_path = merge_objs(
                part['objs'],
                part['name'] + '_' + str(name_counts[part['name']]), objs_path,
                os.path.join(args.dest, object_id), args.path_only)
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

        with open(os.path.join(args.dest, object_id, 'result_merged.json'),
                  'w') as f:
            json.dump(result, f)
