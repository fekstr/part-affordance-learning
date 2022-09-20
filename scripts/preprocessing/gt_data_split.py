import random
import math
from collections import defaultdict
import pandas as pd

gt = pd.read_csv('./data/gt.csv')

# Format dataframe correctly
gt = gt.rename(columns={
    'y>n': 'has_affordance',
    'Unnamed: 17': 'affordance',
    'Unnamed: 18': 'object',
    'Unnamed: 19': 'object_part',
    'Unnamed: 20': 'body_part'
})
gt = gt[['has_affordance', 'affordance', 'object', 'object_part', 'body_part']]
gt = gt[:-3]
gt = gt[gt['has_affordance'] == 't'] # Only use true affordances
gt = gt.reset_index(drop=True)
gt = gt.loc[:, gt.columns != 'has_affordance']

# Delete object classes that do not exist in PartNet v0
classes_to_delete = ['couch', 'cup', 'handbag', 'suitcase']
gt = gt[~gt['object'].isin(classes_to_delete)]
gt.to_pickle('./data/gt.pkl')

# Extract affordance labels to use in the study
occurrences = gt.groupby(['affordance']).nunique()
occurrences = occurrences[occurrences['object'] > 1] # Only use affordances that exist for at least 2 objects
occurrences = occurrences.sort_values(by='object')
occurrences = occurrences.reset_index()
affordances = list(occurrences['affordance'])

with open('./data/affordances.txt', 'w') as f:
    for aff in affordances:
        f.write(aff + '\n')

# Split into train and test (by object)
gt = gt[gt['affordance'].isin(occurrences['affordance'])].reset_index(drop=True)
# Create map from affordances to sets of objects
pairs = list(gt.groupby(['affordance', 'object']).count().index)
affordance_object_map = defaultdict(lambda: set())
for pair in pairs:
    affordance_object_map[pair[0]].add(pair[1])

affordance_object_counts = {affordance: len(objects) for affordance, objects in affordance_object_map.items()}

# Randomly select half of the objects, with the constraint that at least one object
# from every affordance must not be selected.
train_ok = False
test_ok = False
while not (train_ok and test_ok):
    objects = set(gt['object'].unique())
    object_count = len(objects)
    train_objects = set()
    while len(train_objects) < object_count // 2 + 1 and len(objects) > 0:
        obj = random.choice(tuple(objects))
        train_objects.add(obj)
        objects.remove(obj)
        for affordance in affordance_object_map:
            affordance_object_map[affordance].discard(obj)
            if len(affordance_object_map[affordance]) == 1:
                objects.discard(list(affordance_object_map[affordance])[0])

    objects = set(gt['object'].unique())
    test_objects = objects.difference(train_objects)

    # Sanity check that all affordances are represented in train and test
    gt_train = gt[gt['object'].isin(train_objects)]
    gt_test = gt[gt['object'].isin(test_objects)]
    all_affordances = set(gt_train['affordance']).union(set(gt_test['affordance']))
    train_ok = all_affordances == set(gt_train['affordance'])
    test_ok = all_affordances == set(gt_test['affordance'])


# Save the split
with open('./data/train_objects.txt', 'w') as f:
    for obj in train_objects:
        f.write(obj + '\n')
with open('./data/test_objects.txt', 'w') as f:
    for obj in test_objects:
        f.write(obj + '\n')