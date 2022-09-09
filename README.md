# Weak Supervised Learning of Object Part Affordances

## Data

### Ground Truth Data
[Ground truth part affordance dataset](https://docs.google.com/spreadsheets/d/1xx0fMgmg8ux-vzHx994hkOuFUAO7-OiHfe_mcGpfvfA/edit#gid=935702000)

- `affordances.txt` contains all affordances that are afforded by at least 2 distinct objects.
- `train_objects.txt` and `test_objects.txt` contain a train-test split of objects, where every affordance in `affordances.txt` is afforded by at least one object in each of the respective sets.

The training data will be on the following format:

`(object_label, object_point_cloud), (part_label, part_point_cloud, part_affordance)`

TODO
- [ ] Pair objects and their parts with segmented point clouds
- [ ] Implement baseline model  