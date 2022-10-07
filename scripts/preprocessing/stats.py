import os
import json
from collections import defaultdict

from tqdm import tqdm

# Name counts
partnet_path = os.path.join('data', 'PartNet')
selected_path = os.path.join(partnet_path, 'selected_objects')
other_path = os.path.join(partnet_path, 'other_objects')

selected_ids = os.listdir(selected_path)
other_ids = os.listdir(other_path)


def count_names(ids, path):
    name_counts = defaultdict(lambda: 0)
    for id in tqdm(ids):
        result_path = os.path.join(path, id, 'result.json')
        with open(result_path) as f:
            result = json.load(f)
            obj = result[0]
            if len(obj['children']) == 1 and 'children' in obj['children'][0]:
                name = obj['children'][0]['name']
            else:
                name = obj['name']
        name_counts[name] += 1
    return name_counts


selected_name_counts = count_names(selected_ids, selected_path)
other_name_counts = count_names(other_ids, other_path)


def sort_dict(d):
    return {
        k: v
        for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)
    }


selected_name_counts = sort_dict(selected_name_counts)
other_name_counts = sort_dict(other_name_counts)

# Part counts by name


def count_parts(ids, path):
    part_counts = defaultdict(lambda: defaultdict(lambda: 0))
    for id in tqdm(ids):
        result_path = os.path.join(path, id, 'result.json')
        with open(result_path) as f:
            result = json.load(f)
            obj = result[0]
            name = obj['name']
        if len(obj['children']) == 1:
            if 'children' in obj['children'][0]:
                name = obj['children'][0]['name']
                children = obj['children'][0]['children']
            else:
                children = obj['children']
        else:
            children = obj['children']
        for child in children:
            part_name = child['name']
            part_counts[name][part_name] += 1
    return part_counts


selected_part_counts = count_parts(selected_ids, selected_path)
other_part_counts = count_parts(other_ids, other_path)

selected_name_counts.keys()
other_name_counts.keys()

selected_part_counts.keys()
selected_part_counts['desk']
other_part_counts.keys()
other_part_counts['desk']

# select_parts = {
#     'train': {
#         'chair': ['chair_back', 'chair_arm', 'chair_seat', 'chair_base'],
#         'regular_table': ['tabletop', 'table_base'],
#         'cabinet': ['cabinet_frame', 'cabinet_base', 'cabinet_door']
#     },
#     'test': {
#         'chair': ['chair_back', 'chair_arm', 'chair_seat', 'chair_base'],
#     }
# }
