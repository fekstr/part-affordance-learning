from collections import defaultdict
from typing import Tuple
import random
import os
import json
import pickle

from tqdm import tqdm
from pathlib import Path
from pyntcloud import PyntCloud
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader


def get_dataloaders(dataset, small: bool, batch_size: int, load_objects: bool):
    if load_objects:
        obj_ids = [
            object_meta['obj_id'] for object_meta in dataset.object_metas
        ]
    else:  # Load parts
        obj_ids = [part_meta['obj_id'] for part_meta in dataset.part_metas]
    train_sampler, valid_sampler, test_sampler = get_split(obj_ids,
                                                           dataset.id_split,
                                                           small=small)
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=train_sampler)
    valid_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler)
    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=test_sampler)
    return train_loader, valid_loader, test_loader


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