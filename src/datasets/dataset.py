from typing import Literal
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
import open3d as o3d

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
                obj = json.load(f)
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
    for object_class in sorted(test_object_classes):
        bc = bytes(object_class, 'utf-8')
        m.update(bc)
    split_id = m.hexdigest()

    split_path = os.path.join('data', 'splits', f'{split_id}.pkl')

    # Load and return split if already created
    if os.path.isfile(split_path) and not force_new_split:
        print('Using existing split...')
        with open(split_path, 'rb') as f:
            id_split = pickle.load(f)
            all_ids = id_split['train'] + id_split['valid'] + id_split['test']
            return id_split, all_ids

    if test:
        raise FileNotFoundError('No split found')

    # Otherwise, create the split
    print('Creating new split...')
    train_ids, train_labels = _get_ids(objects_path, train_object_classes)
    test_ids, test_labels = _get_ids(objects_path, test_object_classes)
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
        # train_ids, valid_ids = train_test_split(train_ids,
        #                                         stratify=train_labels)
        test_ids, valid_ids = train_test_split(test_ids, stratify=test_labels)
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


def _get_max_parts():
    # max_len = 0
    # for meta in object_metas:
    #     l = len(meta['part_pc_paths'])
    #     if l > max_len:
    #         max_len = l
    # return max_len
    with open('data/PartNet/label_map.json') as f:
        label_map = json.load(f)

    max_len = 0
    for name in label_map:
        parts = label_map[name]
        l = len(parts.keys())
        if l > max_len:
            max_len = l

    return max_len


class CommonDataset(Dataset):
    def __init__(self,
                 objects_path: str,
                 tag: str,
                 num_points: int,
                 num_slots: int,
                 train_object_classes,
                 test_object_classes,
                 affordances,
                 manual_labels=None,
                 item_type=Literal['object', 'labeled_part', 'all_part'],
                 force_new_split=False,
                 test=False,
                 use_cached_metas=False):
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
            objects_path, object_ids, num_points, use_cached=use_cached_metas)
        self.num_points = num_points
        # create_missing_pcs(self.object_metas + self.part_metas, num_points)
        # create_missing_pcs(self.object_metas, num_points)
        self.manual_labels = manual_labels
        self.item_type = item_type
        self.max_parts = _get_max_parts()
        self.num_slots = num_slots

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
        masks = np.zeros((self.max_parts, labels.shape[0]))
        # masks = np.ones((self.max_parts, labels.shape[0])) * float('Inf')
        for label in range(labels.max() + 1):
            mask = (labels == label).astype(int)
            masks[label, :] = mask.squeeze()
        return masks

    def __getitem__(self, idx):
        if self.item_type == 'all_part':
            return self._get_all_parts(idx)
        elif self.item_type == 'labeled_part':
            return self._get_labeled_part(idx)
        elif self.item_type == 'object':
            return self._get_object(idx)
        else:
            raise ValueError(
                'Item type must be one of "all_part", "labeled_part", "object"'
            )

    def _get_object(self, idx):
        # meta = self.object_metas[idx]
        meta = self.object_metas[idx]
        # object_pc = PyntCloud.from_file(meta['pc_path'])
        # object_pc = object_pc.points.to_numpy()
        object_pc = o3d.t.io.read_point_cloud(meta['pc_path'])
        points = object_pc.point['positions'].numpy()
        object_point_cloud = torch.from_numpy(points).T
        # if meta['obj_name'] not in self.seen_objects:
        #     self.seen_objects.append(meta['obj_name'])
        affordance = self._encode_affordances(
            self.manual_labels[meta['obj_name']])

        seg_mask = self._pc_to_seg_mask(object_pc)

        target = {'affordance': affordance, 'segmentation_mask': seg_mask}

        return object_point_cloud, target, {
            'obj_name': meta['obj_name'],
            'obj_id': meta['obj_id']
        }

    def _get_all_parts(self, idx):
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
        part_point_clouds = torch.zeros((self.max_parts, self.num_points, 3))
        for i, part_pc in enumerate(part_pcs):
            pc_tensor = torch.from_numpy(part_pc)
            part_point_clouds[i, :, :] = pc_tensor
        return part_point_clouds, affordance, {}
        # ,{
        #     'obj_id': meta['obj_id'],
        #     'part_names': meta['part_names']
        # }

    def _get_labeled_part(self, idx):
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
