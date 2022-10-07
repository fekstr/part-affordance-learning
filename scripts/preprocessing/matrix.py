import os
import json
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import numpy as np

objects_path = os.path.join('data', 'PartNet', 'selected_objects')
object_ids = os.listdir(objects_path)

name_affordance_map = defaultdict(lambda: set())
for object_id in tqdm(object_ids):
    with open(os.path.join(objects_path, object_id,
                           'result_labeled.json')) as f:
        result_labeled = json.load(f)
        obj = result_labeled[0]
        name = obj['name']
        for part in obj['labeled_parts']:
            name_affordance_map[name].update(part['affordances'])

all_affordances = set()
for object_affordances in name_affordance_map.values():
    all_affordances.update(object_affordances)

all_affordances_list = sorted(list(all_affordances))
affordance_index_map = dict()
for i, affordance in enumerate(all_affordances_list):
    affordance_index_map[affordance] = i

name_affordance_vector_map = dict()
for name, affordance_set in name_affordance_map.items():
    affordance_vector = np.zeros(len(all_affordances), dtype=int)
    for affordance in affordance_set:
        idx = affordance_index_map[affordance]
        affordance_vector[idx] = 1
    name_affordance_vector_map[name] = affordance_vector

df = pd.DataFrame(name_affordance_vector_map, index=all_affordances_list)
df['count'] = df.sum(axis=1)
df[df['count'] > 1].sort_values(by='count', ascending=False)