from typing import List, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import open3d as o3d

from src.datasets.utils import get_metas


class CommonDataset(Dataset):
    def __init__(
        self,
        objects_path: str,
        object_ids: List[str],
        num_points: int,
        num_slots: int,
        affordances,
        object_affordance_labels=None,
    ):
        self.affordances = sorted(affordances)
        self._init_affordance_maps(self.affordances)
        self.num_class = len(affordances)
        self.object_metas, self.part_metas = get_metas(objects_path,
                                                       object_ids,
                                                       num_points,
                                                       use_cached=False)
        self.num_points = num_points
        self.object_affordance_labels = object_affordance_labels
        self.num_slots = num_slots
        self.class_weights = self._get_class_weights()

    def get_small_subset(self, size: int = 16):
        """Returns the indices of a small subset with all classes represented"""
        classes = set()
        for meta in self.object_metas:
            classes.add(meta['obj_name'])
        samples_per_class = size // len(classes)
        counts = defaultdict(lambda: 0)
        idxs = []
        for i, meta in enumerate(self.object_metas):
            if counts[meta['obj_name']] < samples_per_class:
                counts[meta['obj_name']] += 1
                idxs.append(i)
        return idxs

    def _get_class_weights(self):
        counts = defaultdict(lambda: 0)
        for meta in self.object_metas:
            counts[meta['obj_name']] += 1
        weights = {name: 1 / count for name, count in counts.items()}
        meta_weights = [
            weights[meta['obj_name']] for meta in self.object_metas
        ]
        return meta_weights

    def _init_affordance_maps(self, affordances):
        self.affordance_index_map = {
            aff: idx
            for idx, aff in enumerate(affordances)
        }
        self.index_affordance_map = {
            idx: aff
            for idx, aff in enumerate(affordances)
        }

    def _encode_affordances(self, affordances):
        """Creates an affordance tensor based on selected affordances"""
        n_affordances = len(self.affordance_index_map)
        affordance_vectors = torch.zeros((self.num_slots, n_affordances))
        for i, affordance in enumerate(affordances):
            try:
                affordance_vectors[i,
                                   self.affordance_index_map[affordance]] = 1
            except KeyError:
                continue
        return affordance_vectors

    def _pc_to_seg_mask(self, pc):
        labels = pc.point['labels'].numpy()
        return torch.tensor(labels).squeeze()
        # t = torch.zeros((self.num_slots, 1024))
        # for label in range(self.num_slots):
        #     t[label, :] = torch.tensor(labels == label).int().squeeze()
        # return t

    def __len__(self):
        return len(self.object_metas)

    def __getitem__(self, idx):
        return self._get_object(idx)

    def _get_object(self, idx):
        meta = self.object_metas[idx]
        object_pc = o3d.t.io.read_point_cloud(meta['pc_path'])
        points = object_pc.point['positions'].numpy()
        object_point_cloud = torch.from_numpy(points).T
        affordance = self._encode_affordances(
            self.object_affordance_labels[meta['obj_name']])

        seg_mask = self._pc_to_seg_mask(object_pc)

        target = {'affordance': affordance, 'segmentation_mask': seg_mask}

        return object_point_cloud, target, {
            'obj_name': meta['obj_name'],
            'obj_id': meta['obj_id']
        }


def get_datasets(
        config, hyperparams,
        id_split) -> Tuple[CommonDataset, CommonDataset, CommonDataset]:
    common_dataset_args = {
        'objects_path': config['data_path'],
        'num_points': config['num_points'],
        'affordances': config['affordances'],
        'object_affordance_labels': config['labels'],
        'num_slots': hyperparams.num_slots
    }

    train_dataset = CommonDataset(
        **common_dataset_args,
        object_ids=id_split['train'],
    )
    valid_dataset = CommonDataset(
        **common_dataset_args,
        object_ids=id_split['valid'],
    )
    test_dataset = CommonDataset(
        **common_dataset_args,
        object_ids=id_split['test'],
    )

    return train_dataset, valid_dataset, test_dataset
