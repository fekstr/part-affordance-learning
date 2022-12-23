# Part Affordance Learning from Geometry

## Quickstart

- Download [PartNet](https://partnet.cs.stanford.edu/)
- Run [scripts/preprocessing/merge_objs_labeled.py](scripts/preprocessing/merge_objs_labeled.py) to merge object meshes into full parts
- Run [scripts/preprocessing/create_pcs.py](scripts/preprocessing/create_pcs.py) to create point clouds with labels computed from the merged meshes
- Edit [src/config.py](src/config.py) to set the train and test object classes and hyperparameters
- Add a `.env` file to the root directory with the `COMET_API_KEY` variable to log to [Comet.ml](https://www.comet.com/)
- Run [scripts/train.py](scripts/train.py) to train a model. The model to be trained and the loss function are imported and set directly in this file. Checkpoints are saved to `data/checkpoints`.
- Run [scripts/test.py](scripts/test.py) to test a model using a specified checkpoint. The visualization script is written to send visualizations to an external device using [open3d ExternalVisualizer](http://www.open3d.org/docs/release/python_api/open3d.visualization.ExternalVisualizer.html).
