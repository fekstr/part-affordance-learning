from zipfile import ZipFile
import pickle
from tqdm import tqdm

archive = ZipFile('./data/PartNet/data_v0.zip')

with open('./data/object_ids.pkl', 'rb') as f:
    obj_id_map = pickle.load(f)

all_ids = []
for ids in obj_id_map.values():
    all_ids += ids

allowed_prefixes = tuple([f'data_v0/{id}/' for id in all_ids])
exclusions = ['.html', 'parts_render']

for name in tqdm(archive.namelist()):
    if name.startswith(allowed_prefixes) and not any(exclusion in name for exclusion in exclusions):
        archive.extract(name, './data/PartNet/selected_obj')
