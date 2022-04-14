# ParaGen 
  
ParaGen is a PyTorch deep learning framework for parallel sequence generation.
Apart from sequence generation, ParaGen also enhances various NLP tasks, including 
sequence-level classification, extraction and generation.
                     
## Requirements and Installation  

* Install third-party dependent package:
```bash
# Ubuntu
apt-get install libopenmpi-dev libssl-dev openssh-server
# CentOS
yum install openmpi openssl openssh-server
# Conda
conda install -c conda-forge mpi4py
```
* To install ParaGen from source:
```bash  
cd ParaGen
pip install -e .
``` 
* For distributed training on multiple GPUs, you need to make sure `horovod` has been installed.
``` bash
# require CMake to install horovod. (https://cmake.org/install/)
HOROVOD_WITH_PYTORCH=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=${NCCL_ROOT_DIR} pip install horovod
```
* Install lightseq to faster train:
``` bash
pip install lightseq
```

## Getting Started

Before using `ParaGen`, it would be helpful to overview how `ParaGen` works.

`ParaGen` is designed as a `task-oriented` framework, where `task` is regarded as the core of all the codes.
A specific task selects all the components for support itself, such as model architectures, training strategies, dataset, and data processing.
Any component within `ParaGen` can be customized, while the existing modules and methods are used as a plug-in library.

As tasks are considered as the core of `ParaGen`, it works with various `modes`, such as `train`, `evaluate`, `preprocess` and `serve`.
Tasks act differently under different modes, by reorganizing the components without code modification.

Please refer to [examples](examples) for detailed instructions. 

# ParaGen Usage and Contribution

We welcome any experimental algorithms on ParaGen.

- Install ParaGen;
- Create your own paragen-plugin libraries under `third_party`;
- Experiment your own algorithms;
- Write a reproducible shell script;
- Create a merge request and assign reviewers to any of us.

