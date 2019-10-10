## mixmatch-tensorflow2.0
This is an implementation of the research paper [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249) in Python and TensorFlow 2.0. All reported results from training use the WideResNet 28-2 architecture as described in the paper.

### Results
#### CIFAR-10
|  Implementation/Labels  |     250    |     500    |    1000    |    2000    |    4000    | 
| ----------------------- |-----------:|-----------:|-----------:|-----------:|-----------:|
| MixMatch Paper          | 88.92±0.87 | 90.35±0.94 | 92.25±0.32 | 92.97±0.15 | 93.76±0.06 |
| mixmatch-tensorflow2.0  |      88.60 |            |            |            |            |

### Prerequisites
pip installs:
~~~
numpy>=1.17.2
pyyaml>=5.1.2
tensorflow>=2.0
tensorflow-datasets>=1.2.0
tqdm>=4.36.1
~~~

### Training
To run a training session simply run the main.py file from the project directory
~~~
python3 main.py
~~~
To run a training session with non-default hyperparameters you have two options

Option #1, run the training session with command line arguments for the hyperparameters you wish to change:
~~~
python3 main.py --dataset "cifar10" --labelled-examples 250 --learning-rate 0.02
~~~
Option #2, run the training session with a .yaml config file:
~~~
python3 main.py --config-path "configs/cifar10@250.yaml"
~~~
Please note if you use command line arguments and a .yaml config file any overlapping arguments will use the value from the config file instead of the value provided by the command line argument

#### Tensorboard
To write tfevent files for tracking training progress in tensorboard simply run a training session using the tensorboard flag:
~~~
python3 main.py --tensorboard
~~~
All tfevent files are written to a .logs directory under the project directory, so to run tensorboard on the logs written during a training session run the following command from inside the project directory:
~~~
tensorboard --logdir .logs/*
~~~

### Citations
~~~
@misc{berthelot2019mixmatch,
    title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
    author={David Berthelot and Nicholas Carlini and Ian Goodfellow and Nicolas Papernot and Avital Oliver and Colin Raffel},
    year={2019},
    eprint={1905.02249},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
~~~
