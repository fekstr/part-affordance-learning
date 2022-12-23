from collections import defaultdict
from typing import Tuple, List
import random
import os
import json
import pickle
import hashlib

from tqdm import tqdm
from pathlib import Path
from pyntcloud import PyntCloud
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


def get_dataloader(dataset,
                   small: bool,
                   batch_size: int,
                   weighted_sampling=False) -> DataLoader:
    if small:
        sub_indices = dataset.get_small_subset()
        dataset = Subset(dataset, sub_indices)
    sampler = WeightedRandomSampler(
        dataset.class_weights,
        len(dataset)) if weighted_sampling and not small else None
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader


def get_split(
    obj_ids,
    id_split,
    small=False
) -> Tuple[SubsetRandomSampler, SubsetRandomSampler, SubsetRandomSampler]:
    idx_split = dict()
    for split in id_split:
        idx_split[split] = [
            idx for idx, id in enumerate(obj_ids) if id in id_split[split]
        ]

    if small:
        for split in idx_split:
            idx_split[split] = random.sample(idx_split[split], 8)

    return SubsetRandomSampler(idx_split['train']), SubsetRandomSampler(
        idx_split['valid']), SubsetRandomSampler(idx_split['test'])


def get_metas(objects_path, object_ids, num_points, use_cached=False):
    part_metas = []
    object_metas = []
    # TODO: this caching is broken
    if use_cached:
        try:
            with open('cache/metas.pkl', 'rb') as f:
                metas = pickle.load(f)
                print('Using cached metas...')
                return metas['object'], metas['part']
        except:
            pass
    print('Loading metas...')
    for id in tqdm(object_ids):
        result_merged_path = os.path.join(objects_path, id,
                                          'result_merged.json')
        with open(result_merged_path) as f:
            obj = json.load(f)
            parts = obj['parts']
            name = obj['name']

        pc_path = os.path.join(objects_path, id, 'point_clouds')
        full_pc_path = os.path.join(pc_path, f'full_{num_points}.ply')
        if not os.path.isfile(full_pc_path):
            print('Skipping', id)
            continue
        obj_part_metas = []
        for part in parts:
            obj_part_metas.append({
                'part_name':
                part['name'],
                'obj_name':
                name,
                'obj_id':
                id,
                'obj_path':
                part['obj_path'],
                'pc_path':
                os.path.join(pc_path, f'{part["name"]}_{num_points}.ply'),
                'full_pc_path':
                full_pc_path,
                'affordances':
                part['affordances'] if 'affordances' in part else None
            })
        part_metas += obj_part_metas
        object_metas.append({
            'obj_id':
            id,
            'obj_name':
            name,
            'obj_path':
            obj['obj_path'],
            'pc_path':
            full_pc_path,
            'part_pc_paths': [meta['pc_path'] for meta in obj_part_metas],
            'part_names': [meta['part_name'] for meta in obj_part_metas]
        })

    metas = {'object': object_metas, 'part': part_metas}
    with open('cache/metas.pkl', 'wb') as f:
        pickle.dump(metas, f)

    return object_metas, part_metas


def create_missing_pcs(metas, num_points):
    missing_metas = [
        meta for meta in metas if not os.path.isfile(meta['pc_path'])
    ]
    # missing_metas = metas
    if len(missing_metas) > 0:
        print(f'Creating {len(missing_metas)} new point clouds...')
        for meta in tqdm(missing_metas):
            try:
                _create_pc(meta['obj_path'], meta['pc_path'], num_points)
            except:
                with open('./failed.log', 'a') as f:
                    f.write(meta['obj_id'] + '\n')


def _create_pc(obj_path: str, dest_pc_path: str, num_points: int):
    obj = PyntCloud.from_file(obj_path)
    pc = obj.get_sample('mesh_random', n=num_points)
    pc_dir_path = os.path.dirname(dest_pc_path)
    Path(pc_dir_path).mkdir(parents=True, exist_ok=True)
    header = [
        'ply\n', 'format ascii 1.0\n', f'element vertex {num_points}\n',
        'property float x\n', 'property float y\n', 'property float z\n',
        'end_header\n'
    ]
    with open(dest_pc_path, 'w') as f:
        for line in header:
            f.write(line)
        for point in pc.to_numpy():
            point = [str(c) for c in point]
            f.write(' '.join(point) + '\n')


def get_ids(objects_path, object_classes):
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


def create_id_split(objects_path: str, split_path: str,
                    train_object_classes: List[str],
                    test_object_classes: List[str]):
    """Creates a train-validation-test split"""

    print('Creating new split...')
    train_ids, train_labels = get_ids(objects_path, train_object_classes)
    test_ids, test_labels = get_ids(objects_path, test_object_classes)
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

    return id_split


def hash(keys):
    m = hashlib.md5()
    for key in keys:
        bc = bytes(key, 'utf-8')
        m.update(bc)
    split_id = m.hexdigest()
    return split_id


def load_id_split(objects_path,
                  train_object_classes,
                  test_object_classes,
                  force_new_split=False,
                  test=False):
    """Creates a train-validation-test split"""

    # Get unique id for object class combination
    hash_keys = [
        objects_path, 'train', *sorted(train_object_classes), 'test',
        *sorted(test_object_classes)
    ]
    split_id = hash(hash_keys)
    split_path = os.path.join('data', 'splits', f'{split_id}.pkl')

    # Load and return split if already created. Create it otherwise.
    if os.path.isfile(split_path) and not force_new_split:
        print('Using existing split...')
        with open(split_path, 'rb') as f:
            id_split = pickle.load(f)
            return id_split
    elif test:
        raise FileNotFoundError('Must test with an existing split')
    else:
        id_split = create_id_split(objects_path, split_path,
                                   train_object_classes, test_object_classes)

    return id_split
