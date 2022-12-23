import os
import re
from collections import defaultdict
import json
from pathlib import Path

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


def sanity_check(labels, points):
    color_map = {
        0: np.array([1, 0, 0]),
        1: np.array([0, 1, 0]),
        2: np.array([0, 0, 1]),
        3: np.array([0, 0, 0]),
        4: np.array([1, 1, 0]),
        5: np.array([1, 0, 1]),
        6: np.array([0, 1, 1]),
        7: np.array([1, 0.5, 1]),
        8: np.array([0.5, 1, 0.5]),
        9: np.array([0, 1, 0.5]),
    }

    colors = np.zeros((1024, 3))

    for i, label in enumerate(labels):
        color = color_map[int(label.numpy()[0])]
        colors[i, :] = color

    pcdn = o3d.geometry.PointCloud()
    pcdn.points = o3d.utility.Vector3dVector(points)
    pcdn.colors = o3d.utility.Vector3dVector(colors)

    ev = o3d.visualization.ExternalVisualizer()
    ev.set([pcdn])


def generate_point_cloud(path: str, n_points: int, label_map: dict):
    pc_dir = os.path.join(path, 'point_clouds')
    pc_path = os.path.join(pc_dir, f'full_{n_points}.ply')

    if os.path.isfile(pc_path):
        return

    result_path = os.path.join(path, 'result_merged.json')
    with open(result_path) as f:
        result = json.load(f)
        name = result['name']

    obj_dir = os.path.join(path, 'objs')

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

    # sanity_check(labels, points)

    Path(pc_dir).mkdir(parents=True, exist_ok=True)
    o3d.t.io.write_point_cloud(pc_path, pcd, write_ascii=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--src')
    parser.add_argument('--part-labels-path')
    args = parser.parse_args()

    data_path = args.src
    obj_ids = os.listdir(data_path)

    paths = [os.path.join(data_path, obj_id) for obj_id in obj_ids]
    with open(args.part_labels_path) as f:
        label_map = json.load(f)

    for obj_id in tqdm(obj_ids):
        path = os.path.join(data_path, obj_id)
        pc_dir = os.path.join(path, 'point_clouds')
        if os.path.exists(pc_dir):
            print(f'Skipping {obj_id}, point cloud already exists')
            continue
        generate_point_cloud(path, 1024, label_map)
