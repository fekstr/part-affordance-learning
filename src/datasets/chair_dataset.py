from typing import Tuple
import os
import json
import torch
from torch.utils.data import Dataset, SubsetRandomSampler
from pyntcloud import PyntCloud
from tqdm import tqdm
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from scripts.preprocessing.utils import load_split


def create_pc(obj_path: str, dest_pc_path: str, num_points: int):
    obj = PyntCloud.from_file(obj_path)
    pc = obj.get_sample('mesh_random', n=num_points)
    pc_dir_path = os.path.dirname(dest_pc_path)
    Path(pc_dir_path).mkdir(parents=True, exist_ok=True)
    header = [
        'ply\n', 'format ascii 1.0\n', 'element vertex 1000\n',
        'property float x\n', 'property float y\n', 'property float z\n',
        'end_header\n'
    ]
    with open(dest_pc_path, 'w') as f:
        for line in header:
            f.write(line)
        for point in pc.to_numpy():
            point = [str(c) for c in point]
            f.write(' '.join(point) + '\n')


def get_metas(objects_path, object_ids, num_points):
    part_metas = []
    object_metas = []
    for id in object_ids:
        result_labeled_path = os.path.join(objects_path, id,
                                           'result_labeled.json')
        pc_path = os.path.join(objects_path, id, 'point_clouds')
        with open(result_labeled_path) as f:
            result_labeled = json.load(f)
            obj = result_labeled[0]
            parts = obj['labeled_parts']

        full_pc_path = os.path.join(pc_path, f'full_{num_points}.ply')
        object_metas.append({
            'obj_id': id,
            'obj_path': obj['obj_path'],
            'pc_path': full_pc_path,
        })
        for part in parts:
            part_metas.append({
                'part_name':
                part['name'],
                'obj_id':
                id,
                'obj_path':
                part['obj_path'],
                'pc_path':
                os.path.join(pc_path, f'{part["name"]}_{num_points}.ply'),
                'full_pc_path':
                full_pc_path,
                'affordances':
                part['affordances']
            })

    return object_metas, part_metas


def get_affordance_vector(affordances: list, affordance_index_map: dict):
    affordance_vector = np.zeros(len(affordance_index_map))
    for affordance in affordances:
        try:
            affordance_vector[affordance_index_map[affordance]] = 1
        except KeyError:
            continue
    return affordance_vector


def create_missing_pcs(metas, num_points):
    missing_metas = [
        meta for meta in metas if not os.path.isfile(meta['pc_path'])
    ]
    if len(missing_metas) > 0:
        print(f'Creating {len(missing_metas)} new point clouds...')
        for meta in tqdm(missing_metas):
            try:
                create_pc(meta['obj_path'], meta['pc_path'], num_points)
            except:
                with open('./failed.log', 'a') as f:
                    f.write(meta['obj_id'] + '\n')


class ChairDataset(Dataset):
    def __init__(self, objects_path: str, num_points: int):
        self.objects_path = objects_path
        all_object_ids = os.listdir(objects_path)
        self.object_ids = self._filter_ids(all_object_ids)

        object_metas, self.part_metas = get_metas(objects_path,
                                                  self.object_ids, num_points)

        # Create any missing point clouds
        create_missing_pcs(object_metas + self.part_metas, num_points)

        self.num_class = 1

    def __len__(self):
        return len(self.part_metas)

    def __getitem__(self, idx):
        object_pc = PyntCloud.from_file(self.part_metas[idx]['full_pc_path'])
        object_pc = object_pc.points.to_numpy()
        part_pc = PyntCloud.from_file(self.part_metas[idx]['pc_path'])
        part_pc = part_pc.points.to_numpy()
        if self.part_metas[idx]['part_name'] == 'chair_seat':
            affordance = torch.tensor([1]).float()
        else:
            affordance = torch.tensor([0]).float()
        object_point_cloud = torch.from_numpy(object_pc).T
        part_point_cloud = torch.from_numpy(part_pc).T
        return object_point_cloud, part_point_cloud, affordance

    def create_split(
        self
    ) -> Tuple[SubsetRandomSampler, SubsetRandomSampler, SubsetRandomSampler]:
        train_valid_ids, test_ids = train_test_split(self.object_ids,
                                                     test_size=0.1)
        train_ids, valid_ids = train_test_split(train_valid_ids, test_size=0.1)
        train_indices = [
            idx for idx, meta in enumerate(self.part_metas)
            if meta['obj_id'] in train_ids
        ]
        valid_indices = [
            idx for idx, meta in enumerate(self.part_metas)
            if meta['obj_id'] in valid_ids
        ]
        test_indices = [
            idx for idx, meta in enumerate(self.part_metas)
            if meta['obj_id'] in test_ids
        ]
        return SubsetRandomSampler(train_indices), SubsetRandomSampler(
            valid_indices), SubsetRandomSampler(test_indices)

    def _filter_ids(self, ids):
        # Filter out non-chairs
        filtered_ids = []
        for id in ids:
            result_path = os.path.join(self.objects_path, id,
                                       'result_labeled.json')
            with open(result_path) as f:
                obj = json.load(f)[0]
                if obj['name'] == 'chair':
                    filtered_ids.append(id)
        return filtered_ids


if __name__ == '__main__':
    dataset = ChairDataset('./data/PartNet/selected_objects', 1024)
    train_sampler, valid_sampler, test_sampler = dataset.create_split()
    dataset.object_ids