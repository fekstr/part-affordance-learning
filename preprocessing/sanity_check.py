import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

# TODO:
# [ ] Check that all affordances are used
# [ ] Check that all parts are found
# [ ] Check that all object classes are represented
# [ ] Check that instances are reasonably distributed across classes
# [ ] Check that the number of objects seems reasonable

objects_path = './data/PartNet/objects'

gt = pd.read_pickle('./data/gt.pkl')

class_names = gt['object'].unique()
part_names = { class_name: [] for class_name in class_names }
for row in gt.iterrows():
    part_names[row[1]['object']].append(row[1]['object_part'])
affordances = gt['affordance'].unique()

stats = {
    'classes': { class_name: 0 for class_name in class_names },
    'parts': { class_name: { part_name: 0 for part_name in part_names[class_name] } for class_name in class_names },
    'affordances': { affordance: 0 for affordance in affordances }
}
stats['parts']['chair']

for obj_id in tqdm(os.listdir(objects_path)):
    result_labeled_path = f'{objects_path}/{obj_id}/result_labeled.json'
    with open(result_labeled_path, 'r') as f:
        result_labeled = json.load(f)
    obj = result_labeled[0]

    stats['classes'][obj['name']] += 1
    
    obj['labeled_parts']
    for part in obj['labeled_parts']:
        stats['parts'][obj['name']][part['text'].lower()] += 1
        for affordance in part['affordances']:
            stats['affordances'][affordance] += 1

    
with open('./data/PartNet/stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

with open('./data/PartNet/stats.json', 'r') as f:
    stats = json.load(f)