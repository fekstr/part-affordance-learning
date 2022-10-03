import os
import torch
from pyntcloud import PyntCloud
import numpy as np

from scripts.preprocessing.utils import load_split
from src.datasets.utils import get_metas, create_missing_pcs
from src.datasets.common_dataset import CommonDataset


def get_affordance_vector(affordances: list, affordance_index_map: dict):
    affordance_vector = np.zeros(len(affordance_index_map))
    for affordance in affordances:
        try:
            affordance_vector[affordance_index_map[affordance]] = 1
        except KeyError:
            continue
    return affordance_vector


class PartDataset(CommonDataset):
    def __init__(self, objects_path: str, num_points: int):
        super().__init__(objects_path, 'part_dataset', 'part', 16, num_points)

    def _filter_ids(self, ids):
        return ids

    def __len__(self):
        return len(self.part_metas)

    def __getitem__(self, idx):
        meta = self.part_metas[idx]
        object_pc = PyntCloud.from_file(meta['full_pc_path'])
        object_pc = object_pc.points.to_numpy()
        part_pc = PyntCloud.from_file(meta['pc_path'])
        part_pc = part_pc.points.to_numpy()
        affordance_vector = get_affordance_vector(meta['affordances'],
                                                  self.affordance_index_map)
        object_point_cloud = torch.from_numpy(object_pc).T
        part_point_cloud = torch.from_numpy(part_pc).T
        affordances = torch.from_numpy(affordance_vector)
        return object_point_cloud, part_point_cloud, affordances, {
            'obj_id': meta['obj_id'],
            'part_name': meta['part_name']
        }

    def __str__(self):
        return 'part_dataset'


if __name__ == '__main__':
    dataset = PartDataset('./data/PartNet/objects_small', 1024)