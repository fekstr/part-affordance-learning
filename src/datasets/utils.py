from typing import Tuple
import pickle
import random
import os
import json
from tqdm import tqdm
from pathlib import Path
from pyntcloud import PyntCloud
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def get_dataloaders(dataset, small: bool, batch_size: int):
    part_obj_ids = [part_meta['obj_id'] for part_meta in dataset.part_metas]
    train_sampler, valid_sampler, test_sampler = get_split(
        dataset.object_ids,
        part_obj_ids,
        save_path=os.path.join('data', f'{dataset.split_name}_split.pkl'),
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
    object_ids,
    part_obj_ids,
    save_path: str,
    small=False
) -> Tuple[SubsetRandomSampler, SubsetRandomSampler, SubsetRandomSampler]:
    if os.path.isfile(save_path):
        print('Loading data split...')
        with open(save_path, 'rb') as f:
            id_split = pickle.load(f)
    else:
        print('Creating new data split...')
        train_valid_ids, test_ids = train_test_split(object_ids, test_size=0.1)
        train_ids, valid_ids = train_test_split(train_valid_ids, test_size=0.1)
        id_split = {'train': train_ids, 'valid': valid_ids, 'test': test_ids}

        with open(save_path, 'wb') as f:
            pickle.dump(id_split, f)

    idx_split = dict()
    for split in id_split:
        idx_split[split] = [
            idx for idx, id in enumerate(part_obj_ids) if id in id_split[split]
        ]

    if small:
        for split in idx_split:
            idx_split[split] = random.sample(idx_split[split], 5)

    return SubsetRandomSampler(idx_split['train']), SubsetRandomSampler(
        idx_split['valid']), SubsetRandomSampler(idx_split['test'])


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