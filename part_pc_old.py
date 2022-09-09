# Input: object directory
# Output: [(part_point_cloud, part_label)]

import json
from collections import defaultdict
import numpy as np

object_dir = './data/PartNet/data_v0/1095'

# Read points file
with open(f'{object_dir}/point_sample/sample-points-all-pts-label-10000.ply', 'r') as f:
    curr_line = next(f)
    header = [curr_line]
    while curr_line != 'end_header\n':
        curr_line = next(f)
        header.append(curr_line)
    points = []
    for line in f:
        points.append(line)

# Read labels
with open(f'{object_dir}/point_sample/sample-points-all-label-10000.txt', 'r') as f:
    labels = [int(line.strip('\n')) for line in f]
unique_labels = np.unique(labels)

# Create map from labels to points
label_point_map = defaultdict(lambda: list())
for point, label in zip(points, labels):
    label_point_map[label].append(point)

# Map numeric labels to text labels
def get_children_ids(children):
    ids = set()
    for child in children:
        ids.add(child['id'])
        if 'children' in child:
            ids = ids.union(get_children_ids(child['children']))
    return set(ids)

with open(f'{object_dir}/result.json', 'r') as f:
    result = json.load(f)[0]
    # Map top-level parts to all children ids
    flat_parts = []
    for part in result['children']:
        if 'children' in part:
            part_children_ids = get_children_ids(part['children'])
            part['children_ids'] = part_children_ids
            part.pop('children', None)
        else:
            part['children_ids'] = set()
        flat_parts.append(part)

# Merge subparts into top-level parts
parts_with_pc = []
for part in flat_parts:
    points = []
    part_ids = [part['id']] + list(part['children_ids'])
    for part_id in part_ids:
        if part_id in label_point_map:
            points += label_point_map[part_id]
    part['points'] = points
    parts_with_pc.append(part)

# Save

with open('./test.ply', 'w') as f:
    for line in header:
        f.write(line)
    for line in part:
        f.write(line)

###### Test if sampling points works
obj = PyntCloud.from_file(f'{object_dir}/objs/new-3.obj')
obj.add_scalar_field('hsv')
# voxelgrid_id = obj.add_structure('voxelgrid', n_x=32, n_y=32, n_z=32)
# pc = obj.get_sample('voxelgrid_nearest', voxelgrid_id=voxelgrid_id, as_PyntCloud=True)
pc = obj.get_sample('mesh_random', n=1000)

with open('./test.ply', 'w') as f:
    for point in pc.to_numpy():
        point = [str(c) for c in point]
        f.write(' '.join(point) + '\n')