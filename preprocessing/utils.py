def _get_obj_names(objs):
    names = []
    for obj in objs:
        names.append(obj['name'])
        if 'children' in obj:
            names += _get_obj_names(obj['children'])
    return names

def get_obj_name(obj, allowed_names: list):
    names = _get_obj_names([obj])
    candidates = [name for name in names if name in allowed_names]
    return candidates[-1]

def _lines_to_list(lines):
    return [line.replace('\n', '') for line in lines]

def load_split():
    with open('./data/train_objects.txt') as f:
        train_objects = _lines_to_list(f.readlines())
    with open('./data/test_objects.txt') as f:
        test_objects = _lines_to_list(f.readlines())
    with open('./data/affordances.txt') as f:
        affordances = _lines_to_list(f.readlines())
    return train_objects, test_objects, affordances