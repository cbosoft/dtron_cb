# Detectron 2 (CB)

This repo contains utilities and scripts for working with [Detectron 2](https://github.com/facebookresearch/detectron2).



# Experiment Configuration

Machine learning "tests" should be treated like a laboratory experiment, even if it takes place in a computer. To that end a notebook should be kept giving detail enough to re-create the experiments. To facilitate this, experiments are specified using **experiment configuration files** (config files).

A config file is written in a structured format called ["YAML"](https://yaml.org/). A good introduction to YAML is given [here](https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started), but I'll breeze over the basics here. YAML files are essentially composed of a set of keys and values:

```yaml
KEY: VALUE
```

The above is a single node, containing a single value. These can be nested:
```yaml
ROOT:
  FOLDER1: FILE
  FOLDER2:
    - FILE1
    - FILE2     
```
In the above example I used the YAML to describe a simple file structure. This is a good way of thinking about it - we know how folders contain fils and other folders which is similar to how a YAML "node" contains values or other nodes. Also, similar to a file structure, we can specify a path to a particular folder, or to a particular *node* in YAML's case:

```python
ROOT.FOLDER2 = ['FILE1', 'FILE2']
```

The above behaviour is not strictly part of YAML, but of a python library called [`yacs`](https://github.com/rbgirshick/yacs). We use `yacs` to read in yaml files into a form we can easily manipulate in `python`.

Each Detectron experiment is specified using a config file. These files contain information about the model, how it's trained, the data it uses and so on. A config file is shown below:

```yaml
PARENT: "zoo:COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
DATASETS:
  ROOT: "/media/raid/cboyle/datasets"
  NAMES: "PolyS"
DATA:
  AUGMENTATIONS:
    - "T.ResizeShortestEdge(short_edge_length=[640, 672, 704, 736, 768, 800], max_size=1333, sample_style=\'choice\')"
    - "T.RandomFlip()"
    - "T.RandomBrightness(0.9, 1.1)"
    - "T.RandomContrast(0.9, 1.1)"
SOLVER:
  MAX_ITERS: 5000
  WARMUP_ITERS: 500
  CHECKPOINT_PERIOD: 500
  STEPS:
    - 4000
    - 4500
SEED: -1
ACTION: "train"
```

This config file as a lot going on, let's unpack it. First off, it specifies a `PARENT` config file. This sets a file which should be loaded first. Values loaded in the parent config are overwritten by subsequent configs. In this way we can have experiments which "derive" from each other. For example, we could have a config containing information about the data location, and then we could have a number of derived configs each testing different training settings. In this case, the `PARENT` key has the value of a config contained within detectron itself (in its "zoo" - signified at the start of the file path). The model zoo configs can be seen [here](https://github.com/facebookresearch/detectron2/tree/main/configs).

After the `PARENT` key, we have a `DATASET` node which specifies the root folder containing all the dataset information. You shouldn't need to change this from the default. Then it specifies a dataset to use during training, "PolyS". It doesn't need to only be a single dataset, you can specify multiple datasets as a list (like the `AUGMENTATIONS` node).

Next is the `DATA` node which contains information on how the data are processed - namely [augmentations](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/).

Followed by the `SOLVER` node - this defines some very important parameters, see below section on training parameters.

Then we have a `SEED` node - this specifies the seed passed to the random number generator. This is to aid reproducibility. (However, reproducibility is not guaranteed, setting the seed will help.) If you set `SEED` to a negative number, the seed itself will be randomly chosen. The finalised config is saved along with the training results. So that if you want to recreate the experiment later on, you have all the required information (including the seed used) to do so.

Finally, we have an `ACTION` node - this specifies what we want to do. At the moment, only the value `train` is supported. In future, `predict` and `cross_validate` will be implemented.

More information on what can be specified in a config file is given in [dtron_cb/config.py](dtron_cb/config.py) and in [detectron 2 itself](https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py).



# Datasets

The datasets root directory specified by default (and in the config file above) is `/media/raid/cboyle/datasets`. This directory contains the metadata (COCO format json files contianing annotations and file listings) for the datasets. The dataset name is the name of these metadata json files without the extension.

For example, if you annotated images of polystyrene spheres and downloaded the json file and called it `poly_spheres.json`, you would specify the dataset with the name `poly_spheres` in the config file.



# Training settings

The `SOLVER` config node contains a few very important parameters. `MAX_ITERS` is roughly equivalent to number of epochs and defines how long training proceeds. `CHECKPOINT_PERIOD` sets how long to wait between checkpointing the model (i.e. saving the model to disk). The training learning rate is not fixed - it is set by a schedulr. This scheduler starts at a particular learning rate and then decays it in steps. The steps are specified in `STEPS` as iteration numbers. The step values are powers of a parameter `GAMMA` (left to default value of 0.1 in the above config). Before the scheduler kicks in, the learning rate is dropped linearly for a number of iterations given by `WARMUP_ITERS`.

There are many more `SOLVER` paramets given in [detectron](https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py), I just picked out some important ones here.


# Running an experiment

With the configuration set up, we only need to run it. How do we run it?

I would create a new script for each experiment or set of experiments we want to undertake.

```python
# run_polys.py
from dtron_cb import run

if __name__ == '__main__':
    run('experiments/exp_polys.yaml')
```

The above reads in the config file specified and then runs that experiment. As it is a script, we're not limited to running one experiment at a time: we can run loads one after the other.

```python
# run_many_polys.py
from glob import glob
from dtron_cb import run

if __name__ == '__main__':
    for fn in glob('experiments/exp_polys_*.yaml'):
        run(fn)
```

The above will search for any file which matches the pattern "experiments/exp_polys_*.yaml" - where the asterisk can expand to anything (or to nothing).


# Training results

By default, the training results will be put into a folder organised by date: `training_results/YYYY-MM-DD_HH-MM-SS/*`. Contained in that results folder are the original config file, a note of the model structure, a json file containing the training metrics, some metadata in another json file (not very useful - config is better source of information) as well as some plots and (once a checkpoint has been reached) the saved model.