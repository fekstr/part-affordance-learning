import os
import json
import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
from tqdm import tqdm
from pathlib import Path
import numpy as np
from preprocessing.utils import load_split


def create_pc(obj_path: str, dest_pc_path:str, num_points: int):
    obj = PyntCloud.from_file(obj_path)
    pc = obj.get_sample('mesh_random', n=num_points)
    pc_dir_path = os.path.dirname(dest_pc_path)
    Path(pc_dir_path).mkdir(parents=True, exist_ok=True)
    header = [
        'ply\n',
        'format ascii 1.0\n',
        'element vertex 1000\n',
        'property float x\n',
        'property float y\n',
        'property float z\n',
        'end_header\n'
    ]
    with open(dest_pc_path, 'w') as f:
        for line in header:
            f.write(line)
        for point in pc.to_numpy():
            point = [str(c) for c in point]
            f.write(' '.join(point) + '\n')

def get_metas(objects_path, num_points):
    object_ids = os.listdir(objects_path)

    part_metas = []
    object_metas = []
    for id in object_ids:
        result_labeled_path = os.path.join(objects_path, id, 'result_labeled.json')
        pc_path = os.path.join(objects_path, id, 'point_clouds')
        with open(result_labeled_path) as f:
            result_labeled = json.load(f)
            obj = result_labeled[0]
            parts = obj['labeled_parts']

        full_pc_path = os.path.join(pc_path, f'full_{num_points}.ply')
        object_metas.append({
            'obj_path': obj['obj_path'],
            'pc_path': full_pc_path,
        })
        for part in parts:
            part_metas.append({
                'obj_path': part['obj_path'],
                'pc_path': os.path.join(pc_path, f'{part["name"]}_{num_points}.ply'),
                'full_pc_path': full_pc_path,
                'affordances': part['affordances']
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


class PartDataset(Dataset):
    def __init__(self, objects_path: str, num_points: int):
        self.objects_path = objects_path
        object_metas, self.part_metas = get_metas(objects_path, num_points)

        # Create any missing point clouds
        metas = object_metas + self.part_metas
        missing_metas = [meta for meta in metas if not os.path.isfile(meta['pc_path'])]
        if len(missing_metas) > 0:
            print(f'Creating {len(missing_metas)} new point clouds...')
            for meta in tqdm(missing_metas):
                create_pc(meta['obj_path'], meta['pc_path'], num_points)

        # Create map for creating affordance tensors
        _, _, affordances = load_split()
        self.affordance_index_map = { 
            aff: idx for idx, aff in enumerate(sorted(affordances))
        }


    def __len__(self):
        return len(self.part_metas)

    def __getitem__(self, idx):
        object_pc = PyntCloud.from_file(self.part_metas[idx]['full_pc_path'])
        object_pc = object_pc.points.to_numpy()
        part_pc = PyntCloud.from_file(self.part_metas[idx]['pc_path'])
        part_pc = part_pc.points.to_numpy()
        affordance_vector = get_affordance_vector(
            self.part_metas[idx]['affordances'],
            self.affordance_index_map
        )
        sample = {
            'object_point_cloud': torch.from_numpy(object_pc),
            'part_point_cloud': torch.from_numpy(part_pc),
            'affordances': torch.from_numpy(affordance_vector)
        }
        return sample


if __name__ == '__main__':
    dataset = PartDataset('./data/PartNet/objects_small', 1000)