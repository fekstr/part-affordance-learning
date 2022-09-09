import numpy as np
from collections import defaultdict

# Semantics
data = np.load("./data/semantics_medium/automated_graph/3DSceneGraph_Airport.npz", allow_pickle=True)

item = data['output'].item()
item.keys()
item['object'][1].keys()
np.where(item['building']['room_voxel_occupancy']==1)[0]
np.where(item['building']['object_voxel_occupancy']==0)[0]
item['building']['object_voxel_occupancy']

objects = list(item['object'].items())
affordances = defaultdict(lambda: set())
classes = defaultdict(lambda: 0)
for obj in objects:
    cl = obj[1]['class_']
    object_affordances = obj[1]['action_affordance']
    for affordance in object_affordances:
        affordances[cl].add(affordance)
    classes[cl] += 1

classes
affordances['toilet']

for obj in objects:
    cl = obj[1]['class_']
    object_affordances = obj[1]['action_affordance']
    if set(object_affordances) != affordances[cl]:
        print("wtf")
        


'play' in affordances

s = set()
a = ['1', '2']
for aa in a:
    s.add(aa)



item['building']

import os
import pickle
import numpy as np
import open3d as o3d


path = './data/BEHAVE/Date01_Sub01_monitor_hand/t0008.000/monitor/fit01'
obj_path = os.path.join(path, 'monitor_fit.ply')
pc = o3d.io.read_point_cloud(obj_path)
o3d.visualization.draw_geometries([pc])

import json

with open('./data/PartNet/data_v0/1095/result.json', 'r') as f:
    result = json.load(f)

result[0]