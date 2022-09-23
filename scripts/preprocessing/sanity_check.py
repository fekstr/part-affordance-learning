import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from scripts.preprocessing.utils import get_obj_name, load_split

# TODO:
# [ ] Check that all affordances are used
# [ ] Check that all parts are found
# [ ] Check that all object classes are represented
# [ ] Check that instances are reasonably distributed across classes
# [ ] Check that the number of objects seems reasonable

objects_path = './data/PartNet/selected_objects'

gt = pd.read_pickle('./data/gt.pkl')

train_objects, test_objects, affordances = load_split()
class_names = train_objects + test_objects

gt = gt[gt['object'].isin(class_names)]
gt = gt[gt['affordance'].isin(affordances)]

part_names = {class_name: [] for class_name in class_names}
for row in gt.iterrows():
    part_names[row[1]['object']].append(row[1]['object_part'])

stats = {
    'classes': {class_name: 0
                for class_name in class_names},
    'parts': {
        class_name: {part_name: 0
                     for part_name in part_names[class_name]}
        for class_name in class_names
    },
    'affordances': {affordance: 0
                    for affordance in affordances}
}

object_suffixes = []
for dataset in ['train', 'test', 'valid']:
    for obj_id in os.listdir(os.path.join(objects_path, dataset)):
        object_suffixes.append(os.path.join(dataset, obj_id))

for obj_suffix in tqdm(object_suffixes):
    result_labeled_path = f'{objects_path}/{obj_suffix}/result_labeled.json'
    with open(result_labeled_path, 'r') as f:
        result_labeled = json.load(f)
    obj = result_labeled[0]

    obj_name = get_obj_name(obj, class_names)
    stats['classes'][obj_name] += 1

    obj['labeled_parts']
    for part in obj['labeled_parts']:
        if part['text'].lower() in part_names[obj_name]:
            stats['parts'][obj_name][part['text'].lower()] += 1
            for affordance in part['affordances']:
                if affordance in affordances:
                    stats['affordances'][affordance] += 1

num_classes = len([cl for cl, count in stats['classes'].items() if count > 0])
total_obj_instances = sum(stats['classes'].values())
num_obj_instances = stats['classes']
num_labeled_parts = {obj: 0 for obj in stats['parts']}
for obj in num_labeled_parts:
    num_labeled_parts[obj] = sum([ct for ct in stats['parts'][obj].values()])
avg_labeled_parts = {
    obj: round(num_labeled_parts[obj] / num_obj_instances[obj], 2)
    for obj in num_labeled_parts
}
num_instances_with_affordance = stats['affordances']
frac_instances_with_affordance = {
    aff: round(count / total_obj_instances, 2)
    for aff, count in stats['affordances'].items()
}
num_affordances = len(
    [aff for aff, count in stats['affordances'].items() if count > 0])
num_class_with_affordance = {aff: 0 for aff in affordances}
for item in gt.groupby(['affordance', 'object']).count().index:
    aff = item[0]
    num_class_with_affordance[aff] += 1


def sort_dict(d):
    return {
        k: v
        for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)
    }


summary = {
    'Selected object classes':
    class_names,
    'Selected affordances':
    affordances,
    'Number of matched affordances':
    num_affordances,
    'Coverage of selected affordances':
    num_affordances / len(affordances),
    'Number of matched classes':
    num_classes,
    'Coverage of selected classes':
    num_classes / len(class_names),
    'Number of object instances':
    total_obj_instances,
    'Number of object instances by class':
    sort_dict(num_obj_instances),
    'Number of labeled parts by class (total)':
    sort_dict(num_labeled_parts),
    'Number of labeled parts by class (avg per instance)':
    sort_dict(avg_labeled_parts),
    'Number of object classes containing affordance':
    sort_dict(num_class_with_affordance),
    'Number of instances containing affordance':
    sort_dict(num_instances_with_affordance),
    'Fraction of instances containing affordance':
    sort_dict(frac_instances_with_affordance)
}

with open('./data/PartNet/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

with open('./data/PartNet/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

# with open('./data/PartNet/stats.json', 'r') as f:
#     stats = json.load(f)