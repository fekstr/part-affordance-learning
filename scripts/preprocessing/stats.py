import os
import json

from tqdm import tqdm

PARTNET_PATH = os.path.join('data', 'PartNet')
SELECTED_PATH = os.path.join(PARTNET_PATH, 'selected_objects')
SELECTED_IDS = os.listdir(SELECTED_PATH)


def count_nested_parts(part):
    if 'children' not in part:
        return 1
    else:
        return {
            child['name']: count_nested_parts(child)
            for child in part['children']
        }


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


if __name__ == '__main__':
    counts = {}
    for id in tqdm(SELECTED_IDS):
        result_path = os.path.join(SELECTED_PATH, id, 'result.json')
        with open(result_path) as f:
            result = json.load(f)
            obj = result[0]
        add_counts(obj['name'], counts, count_nested_parts(obj))
