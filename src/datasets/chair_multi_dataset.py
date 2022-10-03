import numpy as np

from src.datasets.chair_dataset import ChairDataset


class ChairMultiDataset(ChairDataset):
    def __init__(self, objects_path: str, num_points: int):
        super().__init__(objects_path, num_points, 'chair_dataset',
                         'chair_multi', 16)

    def _get_affordance(self, part_meta):
        affordance_vector = np.zeros(len(self.affordance_index_map))
        for affordance in part_meta['affordances']:
            try:
                affordance_vector[self.affordance_index_map[affordance]] = 1
            except KeyError:
                continue
        return affordance_vector
