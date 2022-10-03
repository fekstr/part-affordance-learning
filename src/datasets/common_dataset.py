import os
import numpy as np
from torch.utils.data import Dataset

from scripts.preprocessing.utils import load_split
from src.datasets.utils import get_metas, create_missing_pcs


class CommonDataset(Dataset):
    def __init__(self, objects_path: str, split_name: str, tag: str,
                 num_class: int, num_points: int):
        self.objects_path = objects_path
        self.object_ids = os.listdir(objects_path)

        _, _, affordances = load_split()
        self.affordance_index_map = {
            aff: idx
            for idx, aff in enumerate(sorted(affordances))
        }
        self.index_affordance_map = {
            idx: aff
            for idx, aff in enumerate(sorted(affordances))
        }
        self.num_class = num_class
        self.split_name = split_name
        self.tag = tag

        self.object_metas, self.part_metas = get_metas(objects_path,
                                                       self.object_ids,
                                                       num_points)

        create_missing_pcs(self.object_metas + self.part_metas, num_points)

    def _get_affordance(self, part_meta):
        affordance_vector = np.zeros(len(self.affordance_index_map))
        for affordance in part_meta['affordances']:
            try:
                affordance_vector[self.affordance_index_map[affordance]] = 1
            except KeyError:
                continue
        return affordance_vector
