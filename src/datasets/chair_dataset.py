import pickle
import os
import json
import torch
from pathlib import Path
from pyntcloud import PyntCloud
from tqdm import tqdm
from pathlib import Path
import numpy as np

from src.datasets.common_dataset import CommonDataset
from src.datasets.utils import get_metas, create_missing_pcs


def get_affordance_vector(affordances: list, affordance_index_map: dict):
    affordance_vector = np.zeros(len(affordance_index_map))
    for affordance in affordances:
        try:
            affordance_vector[affordance_index_map[affordance]] = 1
        except KeyError:
            continue
    return affordance_vector


class ChairDataset(CommonDataset):
    def __init__(self, objects_path: str, num_points: int, split_name: str,
                 tag: str, num_class: int):
        super().__init__(objects_path, split_name, tag, num_class)

        all_object_ids = os.listdir(objects_path)
        self.object_ids = self._filter_ids(all_object_ids)

        object_metas, self.part_metas = get_metas(objects_path,
                                                  self.object_ids, num_points)

        # Create any missing point clouds
        create_missing_pcs(object_metas + self.part_metas, num_points)

    def __len__(self):
        return len(self.part_metas)

    def __getitem__(self, idx):
        meta = self.part_metas[idx]
        object_pc = PyntCloud.from_file(meta['full_pc_path'])
        object_pc = object_pc.points.to_numpy()
        part_pc = PyntCloud.from_file(meta['pc_path'])
        part_pc = part_pc.points.to_numpy()
        affordance = self._get_affordance(meta)
        object_point_cloud = torch.from_numpy(object_pc).T
        part_point_cloud = torch.from_numpy(part_pc).T
        return object_point_cloud, part_point_cloud, affordance, {
            'obj_id': meta['obj_id'],
            'part_name': meta['part_name']
        }

    def _get_affordance(self, part_meta):
        if part_meta['part_name'] in ['chair_seat', 'chair_back']:
            affordance = torch.tensor([1]).float()
        else:
            affordance = torch.tensor([0]).float()
        return affordance

    def _filter_ids(self, ids):
        # Filter out non-chairs
        cache_path = './cache/chair_dataset_ids.pkl'
        if os.path.isfile(cache_path):
            with open(cache_path, 'rb') as f:
                filtered_ids = pickle.load(f)
                return filtered_ids

        filtered_ids = []
        print('Finding chair ids...')
        for id in tqdm(ids):
            result_path = os.path.join(self.objects_path, id,
                                       'result_labeled.json')
            with open(result_path) as f:
                obj = json.load(f)[0]
                if obj['name'] == 'chair':
                    filtered_ids.append(id)
        Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(filtered_ids, f)
        return filtered_ids
