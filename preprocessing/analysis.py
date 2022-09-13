import pickle
from collections import defaultdict

with open('./data/object_ids.pkl', 'rb') as f:
    obj_id_map = pickle.load(f)

# Find ids appearing in multiple classes
id_classes = defaultdict(lambda: list())
for class_name, class_ids in obj_id_map.items():
    for class_id in class_ids:
        id_classes[class_id].append(class_name)

# Intersections
intersections = [tuple(combo) for combo in id_classes.values() if len(combo) > 1]
counts = defaultdict(lambda: 0)
for combo in intersections:
    counts[combo] += 1
