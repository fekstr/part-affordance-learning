import os
import numpy as np
import hashlib
import pickle
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pyntcloud import PyntCloud

from scripts.preprocessing.utils import load_split
from src.datasets.utils import get_metas, create_missing_pcs


def _get_ids(objects_path, object_classes):
    """Gets object ids for selected classes"""
    object_ids = os.listdir(objects_path)

    # Load map from IDs to object names
    name_id_map_path = os.path.join('cache', 'name_id_map.pkl')
    # if os.path.isfile(name_id_map_path):
    if False:
        with open(name_id_map_path, 'rb') as f:
            name_id_map = pickle.load(f)
    else:
        name_id_map = defaultdict(lambda: list())
        for id in tqdm(object_ids):
            # result_path = os.path.join(objects_path, id, 'result_labeled.json')
            result_path = os.path.join(objects_path, id, 'result_merged.json')
            with open(result_path) as f:
                obj = json.load(f)[0]
                name_id_map[obj['name']].append(id)
        with open(name_id_map_path, 'wb') as f:
            pickle.dump(dict(name_id_map), f)

    filtered_ids = []
    labels = []
    for object_class in object_classes:
        class_ids = name_id_map[object_class]
        filtered_ids += class_ids
        labels += [object_class] * len(class_ids)

    return filtered_ids, labels


def _get_split(objects_path,
               train_object_classes,
               test_object_classes,
               force_new_split,
               test=False):
    """Creates a train-validation-test split"""

    # Get unique id for object class combination
    m = hashlib.md5()
    bc = bytes(objects_path, 'utf-8')
    m.update(bc)
    bc = bytes('train', 'utf-8')
    m.update(bc)
    for object_class in sorted(train_object_classes):
        bc = bytes(object_class, 'utf-8')
        m.update(bc)
    bc = bytes('test', 'utf-8')
    m.update(bc)
    for object_class in sorted(train_object_classes + test_object_classes):
        bc = bytes(object_class, 'utf-8')
        m.update(bc)
    split_id = m.hexdigest()

    split_path = os.path.join('data', 'splits', f'{split_id}.pkl')

    # Load and return split if already created
    if os.path.isfile(split_path) and not force_new_split:
        with open(split_path, 'rb') as f:
            id_split = pickle.load(f)
            all_ids = id_split['train'] + id_split['valid'] + id_split['test']
            return id_split, all_ids

    if test:
        raise FileNotFoundError('No split found')

    # Otherwise, create the split
    train_ids, train_labels = _get_ids(objects_path, train_object_classes)
    test_ids, test_labels = _get_ids(objects_path, train_object_classes)
    if train_ids == test_ids:
        all_ids = train_ids
        all_labels = train_labels
    else:
        all_ids = train_ids + test_ids
        all_labels = train_labels + test_labels

    intersection = set(train_object_classes).intersection(
        set(test_object_classes))

    if len(intersection) == len(train_object_classes):
        train_valid_ids, test_ids, train_valid_labels, _ = train_test_split(
            all_ids, all_labels, test_size=0.1, stratify=all_labels)
        train_ids, valid_ids = train_test_split(train_valid_ids,
                                                test_size=0.1,
                                                stratify=train_valid_labels)
    elif len(intersection) == 0:
        train_ids, valid_ids = train_test_split(train_ids,
                                                stratify=train_labels)
    else:
        raise ValueError(
            'Train and test object classes must be equal or disjoint')

    id_split = {'train': train_ids, 'valid': valid_ids, 'test': test_ids}

    # Save
    with open(split_path, 'wb') as f:
        pickle.dump(id_split, f)

    all_ids = train_ids + valid_ids + test_ids

    return id_split, all_ids


def _validate_options(object_classes, affordances):
    train_objects, test_objects, all_affordances = load_split()
    all_objects = train_objects + test_objects

    # Validate affordances
    for affordance in affordances:
        if affordance not in all_affordances:
            raise ValueError('Invalid affordance:', affordance)

    # Validate object classes
    for object_class in object_classes:
        if object_class not in all_objects:
            raise ValueError('Invalid object class:', object_class)


class CommonDataset(Dataset):
    def __init__(self,
                 objects_path: str,
                 tag: str,
                 num_points: int,
                 train_object_classes,
                 test_object_classes,
                 affordances,
                 manual_labels=None,
                 include_unlabeled_parts=False,
                 return_all_parts=False,
                 force_new_split=False,
                 test=False):
        # TODO: add shortcut for using all classes
        object_classes = train_object_classes + test_object_classes
        # _validate_options(object_classes, affordances)
        self.objects_path = objects_path
        self.id_split, object_ids = _get_split(objects_path,
                                               train_object_classes,
                                               test_object_classes,
                                               force_new_split,
                                               test=test)
        self.affordances = sorted(affordances)
        self._init_affordance_maps(self.affordances)
        self.num_class = len(affordances)
        self.tag = tag
        self.object_metas, self.part_metas = get_metas(
            objects_path,
            object_ids,
            num_points,
            include_unlabeled_parts=include_unlabeled_parts)
        create_missing_pcs(self.object_metas + self.part_metas, num_points)
        self.manual_labels = manual_labels
        self.return_all_parts = return_all_parts

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
        affordance_vector = np.zeros(len(self.affordance_index_map))
        for affordance in affordances:
            try:
                affordance_vector[self.affordance_index_map[affordance]] = 1
            except KeyError:
                continue
        return torch.tensor(affordance_vector)

    def __getitem__(self, idx):
        if self.return_all_parts:
            return self._get_parts(idx)
        else:
            return self._get_part(idx)

    def _get_parts(self, idx):
        meta = self.object_metas[idx]
        part_pcs = []
        for path in meta['part_pc_paths']:
            part_pc = PyntCloud.from_file(path)
            part_pc = part_pc.points.to_numpy()
            part_pcs.append(part_pc)
        if self.manual_labels:
            affordance = self._encode_affordances(
                self.manual_labels[meta['obj_name']])
        else:
            affordance = self._encode_affordances(meta['affordances'])
        part_point_clouds = [
            torch.from_numpy(part_pc).T for part_pc in part_pcs
        ]
        # TODO: figure out how to return dynamic size tensors without crashing
        part_point_clouds = []
        for part_pc in part_pcs:
            pc_tensor = torch.from_numpy(part_pc).T.unsqueeze(dim=2)
            part_point_clouds.append(pc_tensor)
        all_pc_tensor = torch.cat(part_point_clouds, dim=2)
        return all_pc_tensor, affordance, {
            'obj_id': meta['obj_id'],
            'part_names': meta['part_names']
        }

    def _get_part(self, idx):
        meta = self.part_metas[idx]
        object_pc = PyntCloud.from_file(meta['full_pc_path'])
        object_pc = object_pc.points.to_numpy()
        part_pc = PyntCloud.from_file(meta['pc_path'])
        part_pc = part_pc.points.to_numpy()
        if self.manual_labels:
            affordance = self._encode_affordances(
                self.manual_labels[meta['obj_name']])
        else:
            affordance = self._encode_affordances(meta['affordances'])
        object_point_cloud = torch.from_numpy(object_pc).T
        part_point_cloud = torch.from_numpy(part_pc).T
        return [object_point_cloud, part_point_cloud], affordance, {
            'obj_id': meta['obj_id'],
            'part_name': meta['part_name']
        }
