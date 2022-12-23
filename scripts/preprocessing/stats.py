import os
import json
from collections import defaultdict

from tqdm import tqdm

# with open('data/PartNet/label_map.json', 'r') as f:
#     label_map = json.load(f)

# labeled_object_names = set(label_map.keys())

partnet_path = os.path.join('data', 'PartNet')
selected_path = os.path.join(partnet_path, 'selected_objects')
selected_ids = os.listdir(selected_path)

# def count_names(ids, path):
#     name_counts = defaultdict(lambda: 0)
#     for id in tqdm(ids):
#         result_path = os.path.join(path, id, 'result.json')
#         with open(result_path) as f:
#             result = json.load(f)
#             obj = result[0]
#             if obj['name'] in labeled_object_names:
#                 name_counts[obj['name']] += 1
#             elif len(obj['children']) == 1 and 'children' in obj['children'][
#                     0] and obj['children'][0]['name'] in labeled_object_names:
#                 name_counts[obj['children'][0]['name']] += 1
#     return name_counts

# def count_parts(ids, path):
#     part_counts = defaultdict(lambda: defaultdict(lambda: 0))
#     for id in tqdm(ids):
#         result_path = os.path.join(path, id, 'result.json')
#         with open(result_path) as f:
#             result = json.load(f)
#             obj = result[0]
#         if obj['name'] in labeled_object_names:
#             name = obj['name']
#         elif len(obj['children']) == 1 and 'children' in obj['children'][0]:
#             name = obj['children'][0]['name']
#         else:
#             continue

#         if name in labeled_object_names:
#             labeled_part_names = set(label_map[name].keys())
#         else:
#             continue

#         if len(obj['children']) == 1:
#             if 'children' in obj['children'][0]:
#                 name = obj['children'][0]['name']
#                 children = obj['children'][0]['children']
#             else:
#                 children = obj['children']
#         else:
#             children = obj['children']
#         for child in children:
#             part_name = child['name']
#             if part_name in labeled_part_names:
#                 part_counts[name][part_name] += 1
#     return part_counts


def count_nested_parts(part):
    if 'children' not in part:
        return 1
    else:
        return {
            child['name']: count_nested_parts(child)
            for child in part['children']
        }


# def sort_dict(d):
#     return {
#         k: v
#         for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)
#     }

counts = {}


def add_counts(name, counts, nested_counts):
    if type(nested_counts) is int:
        if name not in counts:
            counts[name] = {'count': 0}
        counts[name]['count'] += nested_counts
    elif type(nested_counts) is dict:
        for k, v in nested_counts.items():
            if name not in counts:
                counts[name] = {'count': 0, 'children': {}}
            add_counts(k, counts[name]['children'], v)
        counts[name]['count'] += 1


for id in tqdm(selected_ids):
    result_path = os.path.join(selected_path, id, 'result.json')
    with open(result_path) as f:
        result = json.load(f)
        obj = result[0]
    add_counts(obj['name'], counts, count_nested_parts(obj))

counts

# counts

# selected_name_counts = count_names(selected_ids, selected_path)
# selected_name_counts = sort_dict(selected_name_counts)
# selected_name_counts
# selected_part_counts = count_parts(selected_ids, selected_path)
# dict({k: sort_dict(dict(inner)) for k, inner in selected_part_counts.items()})
# nested_parts = count_nested_parts(selected_ids, selected_path)


def get_depth(tree):
    max_depth = 0
    for subtree in tree.values():
        depth = 1
        if type(subtree) is dict:
            depth += get_depth(subtree)
        if depth > max_depth:
            max_depth = depth
    return max_depth


with open('scripts/preprocessing/whatever.json', 'r') as f:
    counts = json.load(f)

# def sum_at(tree, depth, sum_depth):
#     if depth == sum_depth:
#         for k, sub in tree.items():
#             if type(sub) is dict:
#                 tree[k] = sum([v for v in sub.values()])
#         return
#     for sub in tree.values():
#         if type(sub) is dict:
#             sum_at(sub, depth + 1, sum_depth)


def delete_level(tree, depth, del_depth):
    if depth == del_depth:
        del tree['children']
        return
    for sub in tree['children'].values():
        if 'children' in sub:
            delete_level(sub, depth + 1, del_depth)


delete_level(counts['cutting_instrument'], 1, 3)
counts['cutting_instrument']
counts
counts.keys()
get_depth(counts['scissors'])


def merge_lowest(path, name):
    with open(path, 'r') as f:
        counts = json.load(f)
        obj_counts = counts[name]
    depth = get_depth(obj_counts)


get_depth(counts['chair'])
counts['chair']

#     depth = 0
#     while

# merge_lowest('scripts/preprocessing/whatever.json', 'chair')
