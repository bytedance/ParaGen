# Antibody Language Models


Our implementation is based on [ParaGen](https://github.com/bytedance/ParaGen). 
To install ParaGen from source:

```bash  
cd ParaGen
pip install -e .
``` 
More information can be found in the official repo. 

We put the source code under the *examples/antibody* directory. It contains the modified [**AbLang**](https://github.com/oxpig/AbLang) and [**ESM**](https://github.com/facebookresearch/esm) to fit this framework. Some of our code was borrowed from them. Thanks for their great work!


The file organization is as follows:

```
ParaGen/examples/antibody
├── ablang                              # AbLang-H / AbLang-L
│   ├── configs     
│   ├── src      
│   └── vocab
├── esm                                 # ESM / MSA-Transformer 
│   ├── configs
│   └── src  
├── src                                 # EATLM
│   ├── models
│   │   ├── axial_transformer_layer.py
│   │   ├── axial_attention.py
|   │   ├── transformer_germline_encoder.py
|   │   └── germline_model.py
│   └── tasks      
│       ├── germline_masked_lm_task.py
|       ├── germline_pretrained_task.py
|       └── germline_task.py 
├── configs                             # EATLM config files     
├── utils                              
│   ├── analyze.py
│   ├── preprocess.py
│   ├── draw.ipynb
│   └── metric.py      
├── scripts                             
│   ├── cross_valid.sh
│   ├── disease_diagnosis.sh
│   └── individual.sh    
└── README.md                           # Readme 
```  


## Finetune pre-trained model for specific tasks

The ParaGen has many options for training and inference. Usually we use a *\*.yaml* file to organize these arguments. These arguments can also be fed to the framework via the command line. 

**\[Attention\] If an option is set more than once, the last value will overwrite the previous one. And the command arguments have a higher priority than the *\*.yaml* configuration files.**

We provide several sample configuration files for each model and task with default hyperparameters. These files can be found under the *configs/* directory of each model. If you want to use these configuration files, please change the options as you need. For example, change the default path.

The antibody tasks are defined under *examples/antibody/src/tasks/*. Therefore, the *examples/antibody/src/* should be added as user-specific library. 

### EATLM

To run with our EATLM, the command looks like this:

``` shell
$ paragen-run --config examples/antibody/configs/downstream/cell_germ.yaml --lib examples/antibody/src
```


### Baseline

For esm and ablang, the specific library should be added: 

``` shell
# esm
$ paragen-run --config examples/antibody/esm/configs/cell.yaml --lib examples/antibody/src,examples/antibody/esm/src

# ablang
$ paragen-run --config examples/antibody/ablang/configs/cell.yaml --lib examples/antibody/src,examples/antibody/ablang/src
```


## Running scripts

We provide some scripts for repeative experiments under the *examples/antibody/scripts* directory. We can use these scripts for k-cross validation.

``` shell
$ bash examples/antibody/scripts/cross_valid.sh <config> <model> <data> k [save/drop]
```

The details explanations of each option can be found in the *cross_valid.sh*. 

Besides, we also provide the postprocessing script for disease diagnosis. The entry is *disease_diagnosis.sh*. It can download sequence-level results from HDFS and calculate the individual-level results:

``` shell
$ bash examples/antibody/scripts/disease_diagnosis.sh [download/seq/ind] <disease> <model> <name>
```

**[Attention] Please modify the default HDFS path before using these scripts.**





