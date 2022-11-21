import os
import re
from collections import defaultdict
import json

import open3d as o3d
import numpy as np
from tqdm import tqdm


def create_label_map(paths: list):
    label_counts = defaultdict(lambda: defaultdict(lambda: 0))
    for path in tqdm(paths):
        result_path = os.path.join(path, 'result_merged.json')
        with open(result_path) as f:
            result = json.load(f)
            name = result['name']

        obj_dir = os.path.join(path, 'objs', 'merged_crude')
        obj_names = os.listdir(obj_dir)
        obj_names.remove('full_0.obj')

        labels = [
            re.sub(r'_[0-9]+\.obj', '', obj_name) for obj_name in obj_names
        ]

        for label in labels:
            label_counts[name][label] += 1

    label_sets = {
        name: {k
               for k, v in label_counts[name].items() if v > 1}
        for name in label_counts.keys()
    }
    label_map = {
        name: {k: i
               for i, k in enumerate(label_sets[name])}
        for name in label_sets
    }
    with open('data/PartNet/label_map.json', 'w') as f:
        json.dump(label_map, f, indent=4)


def generate_point_cloud(path: str, n_points: int, label_map: dict):
    pc_path = os.path.join(path, 'point_clouds', f'full_{n_points}.ply')

    if os.path.isfile(pc_path):
        return

    result_path = os.path.join(path, 'result_merged.json')
    with open(result_path) as f:
        result = json.load(f)
        name = result['name']

    obj_dir = os.path.join(path, 'objs', 'merged_crude')

    full_mesh_path = os.path.join(obj_dir, 'full_0.obj')
    full_mesh = o3d.io.read_triangle_mesh(full_mesh_path)
    pcd = full_mesh.sample_points_uniformly(number_of_points=n_points)

    # Create scene
    obj_names = os.listdir(obj_dir)
    obj_names.remove('full_0.obj')

    if len(obj_names) == 0:
        return

    scene = o3d.t.geometry.RaycastingScene()

    mesh_ids = {}
    label_count = 0
    for obj_name in obj_names:
        obj_path = os.path.join(obj_dir, obj_name)
        mesh = o3d.io.read_triangle_mesh(obj_path)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        label = re.sub(r'_.\.obj', '', obj_name)

        if label not in label_map[name]:
            continue
        # if name not in label_map:
        #     label_map[name] = {}
        # if label not in label_map[name]:
        #     if len(label_map[name].values()) == 0:
        #         label_map[name][label] = 0
        #     else:
        #         max_id = max(label_map[name].values())
        #         label_map[name][label] = max_id + 1

        label_count += 1
        mesh_ids[scene.add_triangles(mesh)] = label

    if label_count == 0:
        return

    points = np.asarray(pcd.points)
    query = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)

    labels = scene.compute_closest_points(query)['geometry_ids']
    labels = labels.numpy()
    labels = np.array([label_map[name][mesh_ids[l]] for l in labels])
    labels = np.expand_dims(labels, 1)
    labels = o3d.core.Tensor(labels, dtype=o3d.core.Dtype.UInt8)

    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    pcd.point['labels'] = labels

    o3d.t.io.write_point_cloud(pc_path, pcd, write_ascii=True)


data_path = 'data/PartNet/selected_objects'
obj_ids = os.listdir(data_path)

paths = [os.path.join(data_path, obj_id) for obj_id in obj_ids]
try:
    with open('data/PartNet/label_map.json') as f:
        label_map = json.load(f)
except FileNotFoundError:
    create_label_map(paths)

for obj_id in tqdm(obj_ids):
    # print(obj_id)
    path = os.path.join(data_path, obj_id)

    # pc_dir_path = os.path.join(path, 'point_clouds')
    # files = os.listdir(pc_dir_path)
    # for file in files:
    #     file_path = os.path.join(pc_dir_path, file)
    #     os.remove(file_path)

    generate_point_cloud(path, 1024, label_map)

# Sanity check
# color_map = {
#     0: np.array([1, 0, 0]),
#     1: np.array([0, 1, 0]),
#     2: np.array([0, 0, 1]),
#     3: np.array([0, 0, 0]),
#     4: np.array([0, 1, 0]),
# }

# colors = np.zeros((1024, 3))

# for i, label in enumerate(labels):
#     color = color_map[label.item()]
#     colors[i, :] = color

# color_vector = o3d.utility.Vector3dVector(colors)
# pcd.colors = color_vector

# o3d.io.write_point_cloud('test.ply', pcd, write_ascii=True)

# ev = o3d.visualization.ExternalVisualizer()
# ev.set([pcd])
