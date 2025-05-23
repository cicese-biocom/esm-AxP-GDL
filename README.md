[![Made with Python](https://img.shields.io/badge/Python-3.7-blue?logo=python&logoColor=white)](https://python.org "Go to Python homepage")
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/docs/1.12/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3.1-%237732a8.svg?style=flat&logo=PyG&logoColor=white)](https://pytorch-geometric.readthedocs.io/en/2.3.1/)
[![CUDA](https://img.shields.io/badge/CUDA-11-%2376B900.svg?style=flat&logo=NVIDIA&logoColor=white)](https://developer.nvidia.com/cuda-11-3-1-download-archive)
[![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

# **esm-AxP-GDL**

esm-AxP-GDL is a framework to build Graph Deep Learning (GDL)-based models leveraging ESMFold-predicted peptide 
structures and ESM-2 based amino acid-level characteristics for the prediction of antimicrobial peptides (AMPs). 
This framework was designed to be easily extended to modeling any task related to the prediction of peptide and 
protein biological activities (or properties).

![workflow_framework](https://github.com/cicese-biocom/esm-AxP-GDL/assets/136017848/99191e5d-d1a5-470b-a905-126bf96e307f)

## **Install esm-AxP-GDL**
Clone the repository:
```
git clone https://github.com/cicese-biocom/esm-AxP-GDL.git
```
The directory structure of the framework is as follows:
```
esm-AxP-GDL
├── best_models                                 <- Best models created using this framework.
│   ├── AMPDiscover                                <- Best models created on the AMPDiscover benchmarking dataset.
│   │   ├── amp_esmt36_d10_hd128_(Model3)                    
│   │   │   ├── Metrics.txt                           <- Matthew correlation coefficient (MCC) achieved by this model. 
│   │   │   ├── Parameters.json                       <- Parameters used to build the model.
│   │   ├── amp_esmt33_d10_hd128_(Model2)                     
│   │   │   ├── Metrics.txt                           <- Matthew correlation coefficient (MCC) achieved by this model. 
│   │   │   ├── Parameters.json                       <- Parameters used to build the model.
│   │   ├── amp_esmt30_d15_hd128_(Model5)                      
│   │   │   ├── Metrics.txt                           <- Matthew correlation coefficient (MCC) achieved by this model. 
│   │   │   ├── Parameters.json                       <- Parameters used to build the model.
├── datasets                                    
│   ├── AMPDiscover                             <- AMPDiscover benchmarking dataset.     
│   │   ├── AMPDiscover.csv                        <- Training, validation and test sets.              
│   │   ├── Test(reduced-100).csv                  <- Reduced test set comprised of sequences of up to 100 amino acids.                
│   │   ├── Test(reduced-30).csv                   <- Reduced test set comprised of sequences of up to 30 amino acids.
│   │   ├── External.csv                           <- External test set containing non-duplicated sequences with the AMPDiscover set.
├── example                          
│   │   ├── ExampleDataset.csv                  <- Example set to run the framework.
├── graph                                        
│   ├── _init_.py                               <- Module to build graphs.
│   ├── construct_graphs.py                     <- Script to build graphs.
│   ├── nodes.py                                <- Script to build the graph nodes.
│   ├── edges.py                                <- Script to build the graph edges.
│   ├── edge_construction_functions.py          <- Classes to build graph edges according to different criteria.               
│   ├── tertiary_structure_handler.py           <- Script to load or predict tertiary structures.
├── workflow                                        
│   ├── _init_.py                               <- Framework workflow module.
│   ├── application_context.py                  <- Classes to handle dependencies.
│   ├── args_parser_handler.py                  <- Classes to parse input arguments.
│   ├── parameters_setter.py                    <- Classes to configure execution parameters.
│   ├── path_creator.py                         <- Classes to create the workflow output path.
│   ├── logging_handler.py                      <- Classes to handle event logger. 
│   ├── data_loader.py                          <- Classes to load input data.
│   ├── dataset_validator.py                    <- Classes to validate input data.
│   ├── gdl_workflow.py                         <- Classes to run the workflow in training, test and inference modes.
│   ├── classification_metrics.py               <- Classes to calculate performance metrics.
├── models                                      
│   ├── _init_.py                               
│   ├── esm2                                    
│   │   ├── _init_.py                           <- ESM-2 module.
│   │   ├── esm2_model_handler.py               <- Script to use a given ESM-2 model.
│   │   ├── checkpoints                         <- Directory where the ESM-2 models are downloaded.
│   ├── esmfold                                 
│   │   ├── _init_.py                           <- ESMFold module.
│   │   ├── esmfold_handler.py                  <- Script to use the ESMFold model.
│   │   ├── checkpoints                         <- Directory where the ESMFold model is downloaded.
│   ├── GAT                                     
│   │   ├── _init_.py                           <- Graph Attention Network (GAT) module.
│   │   ├── GAT.py                              <- Script to use the implemented GAT architecture.
├── utils                                                             
│   │   ├── _init_.py                           
│   │   ├── distances.py                        <- Calculates the distance between atom-pairs using different functions.
│   │   ├── pdb_parser.py                       <- Parses files in PDB format.
│   │   ├── json_parser.py                      <- Parses files in JSON format.
│   │   ├── file_system_handler.py              <- Manages file system operations               
├── settings
│   │   ├── _init_.py                           <- Settings module
│   │   ├── esm2_representation.json            <- ESM-2 representation that can be used.
│   │   ├── logger_setting.json                 <- Configuration of the event logger.
│   │   ├── output_settings.json                <- Output file system configuration.
├── README.md                                   <- README
├── environment.yml                             <- Python libraries required.
├── Dockerfile                                  <- Docker image.
├── docker-compose.yml                          <- Configuration of the Docker container.
├── train.py                                    <- Script to train a model.
├── test.py                                     <- Script to test a model.
├── inference.py                                <- Script to use a model for inference.
├── train.sh                                    <- Example script for training.
├── test.sh                                     <- Example script for test.
├── inference.sh                                <- Example to use a model for inference.
```

## **Dependencies**
This framework is currently supported for Linux, Python 3.7, CUDA 11 and Pytorch 1.12. The major dependencies used in this project are:

>C++ compiler: https://gcc.gnu.org/ </br>
CUDA Toolkit: https://developer.nvidia.com/ </br>
Python: 3.7 </br>
PyTorch: 1.12.0+cu113 </br>
PyTorch Geometric: (torch-cluster: 1.6.1, torch-scatter: 2.1.0, torch-sparse: 0.6.15, torch-geometric: 2.3.1) </br>
ESM-2 (fair-esm:2.0.0) </br> 
ESMFold (fair-esm:2.0.0) 

The Python libraries used in the workflow are specified in the `environment.yml` file.

### **Python environment configuration via conda**
We provide the steps to create a Python environment from an `environment.yml` file using conda:
```
1. conda env create -f environment.yml
2. conda activate esm-axp-gdl-env
3. conda env list
``` 

### **Managing dependencies using Docker container**
We provide the `Dockerfile` and `docker-compose.yml` files with all the dependencies and configurations required by the framework.
#### Prerequisites:
1. Install Docker following the installation guidelines for your platform: https://docs.docker.com/engine/installation/
2. Install CUDA Toolkit: https://developer.nvidia.com/

#### Build the Docker image locally from the next command line:
```
docker-compose build
```

NOTE: if a docker image is used to run this framework, then the path of the input files should be relative to
the framework directory.

## **Install on computer clusters**
The installation on computer clusters depends on the applications available to users through modular environment commands. 
An installation example could be:
```
1. module load python/ondemand-jupyter-python3.8
2. module load gcc/9.2.0
3. module load cuda/11.3.0
4. conda env create -f environment.yml
```

NOTE: we provide template scripts to run training/test/inference Slurm batch jobs.

## **Usage**
### **Input data format**
The framework esm-AxP-GDL is inputted with a comma separated value (CSV) file, which contains 
the identifier, the amino acid sequence, the activity value, and the partition of each peptide. 
We used the numbers 1, 2 and 3 to represent the training, validation, and test sets, respectively. 
For training or using a model for inference, it should be specified the path for the input CSV file.

### **For training or using a model for inference**
train.py and test.py are used to carry out the training and inference steps, respectively. 
The next command lines can be used to run the training and inference steps, respectively.

#### Train
```
usage: train.py [-h] --dataset DATASET [--tertiary_structure_method {esmfold}]
                [--pdb_path PDB_PATH] [--batch_size BATCH_SIZE]
                --gdl_model_path GDL_MODEL_PATH
                [--esm2_representation {esm2_t6,esm2_t12,esm2_t30,esm2_t33,esm2_t36,esm2_t48}]
                [--edge_construction_functions EDGE_CONSTRUCTION_FUNCTIONS]
                [--distance_function {euclidean,canberra,lance_williams,clark,soergel,bhattacharyya,angular_separation}]
                [--distance_threshold DISTANCE_THRESHOLD]
                [--amino_acid_representation {CA}]
                [--number_of_heads NUMBER_OF_HEADS]
                [--hidden_layer_dimension HIDDEN_LAYER_DIMENSION]
                [--add_self_loops] [--use_edge_attr]
                [--learning_rate LEARNING_RATE] [--dropout_rate DROPOUT_RATE]
                [--number_of_epochs NUMBER_OF_EPOCHS] [--save_ckpt_per_epoch]
                [--validation_mode {random_coordinates,random_embeddings}]
                [--randomness_percentage RANDOMNESS_PERCENTAGE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the input dataset in CSV format
  --tertiary_structure_method {esmfold}
                        3D structure prediction method. None indicates to load
                        existing tertiary structures from PDB files ,
                        otherwise, sequences in input CSV file are predicted
                        using the specified method
  --pdb_path PDB_PATH   Path where tertiary structures are saved in or loaded
                        from PDB files
  --batch_size BATCH_SIZE
                        Batch size
  --gdl_model_path GDL_MODEL_PATH
                        The path to save/load the models
  --esm2_representation {esm2_t6,esm2_t12,esm2_t30,esm2_t33,esm2_t36,esm2_t48, 'reduced_esm2_t6', 'reduced_esm2_t12',
                        'reduced_esm2_t30', 'reduced_esm2_t33', 'reduced_esm2_t36', 'combined_esm2'}
                        ESM-2 representation to be used
  --edge_construction_functions EDGE_CONSTRUCTION_FUNCTIONS
                        Criteria (e.g., distance) to define a relationship
                        (graph edges) between amino acids. Only one ESM-2
                        contact map can be specified. The options available
                        are: 'distance_based_threshold', 'sequence_based',
                        'esm2_contact_map_50', 'esm2_contact_map_60',
                        'esm2_contact_map_70', 'esm2_contact_map_80',
                        'esm2_contact_map_90'
  --distance_function {euclidean,canberra,lance_williams,clark,soergel,bhattacharyya,angular_separation}
                        Distance function to construct graph edges
  --distance_threshold DISTANCE_THRESHOLD
                        Distance threshold to construct graph edges
  --amino_acid_representation {CA}
                        Reference atom into an amino acid to define a
                        relationship (e.g., distance) regarding another amino
                        acid
  --number_of_heads NUMBER_OF_HEADS
                        Number of heads
  --hidden_layer_dimension HIDDEN_LAYER_DIMENSION
                        Hidden layer dimension
  --add_self_loops      True if specified, otherwise, False. True indicates to
                        use auto loops in attention layer.
  --use_edge_attr       True if specified, otherwise, False. True indicates to
                        use edge attributes in graph learning.
  --learning_rate LEARNING_RATE
                        Learning rate
  --dropout_rate DROPOUT_RATE
                        Dropout rate
  --number_of_epochs NUMBER_OF_EPOCHS
                        Maximum number of epochs
  --save_ckpt_per_epoch
                        True if specified, otherwise, False. True indicates
                        that the models of every epoch will be saved. False
                        indicates that the latest model and the best model
                        regarding the MCC metric will be saved.
  --validation_mode {random_coordinates,random_embeddings}
                        Criteria to corroborate that the predictions of the
                        models are not by chance
  --randomness_percentage RANDOMNESS_PERCENTAGE
                        Percentage of rows to be randomly generated     
```

#### Test
```
usage: test.py [-h] --dataset DATASET [--tertiary_structure_method {esmfold}]
               [--pdb_path PDB_PATH] [--batch_size BATCH_SIZE]
               --gdl_model_path GDL_MODEL_PATH [--dropout_rate DROPOUT_RATE]
               --output_path OUTPUT_PATH [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the input dataset in CSV format
  --tertiary_structure_method {esmfold}
                        3D structure prediction method. None indicates to load
                        existing tertiary structures from PDB files ,
                        otherwise, sequences in input CSV file are predicted
                        using the specified method
  --pdb_path PDB_PATH   Path where tertiary structures are saved in or loaded
                        from PDB files
  --batch_size BATCH_SIZE
                        Batch size
  --gdl_model_path GDL_MODEL_PATH
                        The path to load the model
  --output_path OUTPUT_PATH
                        The path where the output data will be saved.
  --seed SEED           Seed to run the test                                          
```
#### Inference
```
usage: inference.py [-h] --dataset DATASET
                    [--tertiary_structure_method {esmfold}]
                    [--pdb_path PDB_PATH] [--batch_size BATCH_SIZE]
                    --gdl_model_path GDL_MODEL_PATH
                    [--dropout_rate DROPOUT_RATE] --output_path OUTPUT_PATH
                    [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the input dataset in CSV format
  --tertiary_structure_method {esmfold}
                        3D structure prediction method. None indicates to load
                        existing tertiary structures from PDB files ,
                        otherwise, sequences in input CSV file are predicted
                        using the specified method
  --pdb_path PDB_PATH   Path where tertiary structures are saved in or loaded
                        from PDB files
  --batch_size BATCH_SIZE
                        Batch size
  --gdl_model_path GDL_MODEL_PATH
                        The path to load the model
  --output_path OUTPUT_PATH
                        The path where the output data will be saved.
  --seed SEED           Seed to run the Inference
                                                
```

### **Example**
We provide the train.sh and test.sh example scripts to train or use a model for inference, respectively.
In these scripts are used the AMPDiscover dataset as input set, the model `esm2_t36_3B_UR50D` to evolutionary 
characterize the graph nodes, a `distance threshold equal to 10 angstroms`
to build the graph edges, and a `hidden layer size equal to 128`.

When using the Docker container, the example scripts should be used as follows:
```
docker-compose run --rm esm-axp-gdl-env sh train.sh
```
```
docker-compose run --rm esm-axp-gdl-env sh test.sh
```
```
docker-compose run --rm esm-axp-gdl-env sh inference.sh
```

### **Best models**
Best models created. So far, the only models are to predict general-AMP.  

| Name                                                                                                                   | Dataset                                                          | Endpoint     | MCC    | Description                                                                                                                                                                                                                                                        |
|------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|--------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [amp_esmt33_d10_hd128_(Model2).pt](https://drive.google.com/file/d/1mskGXsYz5yjNxQUoJwRWDHi_it1bORoG/view?usp=sharing) | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9389 | This model was created using the AMPDiscover dataset as input data, the model `esm2_t33_650M_UR50D` to evolutionarily characterize the graph nodes, a `distance threshold equal to 10 angstroms` to build the graph edges, and a `hidden layer size equal to 128`. |
| [amp_esmt36_d10_hd128_(Model3).pt](https://drive.google.com/file/d/1pBkNn6-_6w5YO2TljMkOVo5xVivnAQAf/view?usp=sharing) | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9505 | This model was created using the AMPDiscover dataset as input data, the model `esm2_t36_3B_UR50D` to evolutionarily characterize the graph nodes, a `distance threshold equal to 10 angstroms` to build the graph edges, and a `hidden layer size equal to 128`.   |
| [amp_esmt30_d15_hd128_(Model5).pt](https://drive.google.com/file/d/1gvGDVTCQ0QmTP6rU9tSBC9rc1e4BV-M-/view?usp=sharing) | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9379 | This model was created using the AMPDiscover dataset as input data, the model `esm2_t30_150M_UR50D` to evolutionarily characterize the graph nodes, a `distance threshold equal to 15 angstroms` to build the graph edges, and a `hidden layer size equal to 128`. |

NOTE:  The performance `metrics` obtained and `parameters` used to build the best models are available at `/best_models` directory. The models are available-freely making click on the Table.