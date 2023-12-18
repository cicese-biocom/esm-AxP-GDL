# **esm-AxP-GDL**

esm-AxP-GDL is a framework to build Graph Deep Learning (GDL)-based models leveraging ESMFold-predicted peptide 
structures and ESM-2 models-based amino acid featurization for the prediction of antimicrobial peptides (AMPs).
This framework was designed to be easily extended to any modeling task related to the prediction of 
peptide and protein biological activities (or properties).

## **Install esm-AxP-GDL**
Clone the repository:
```
git clone https://github.com/cicese-biocom/esm-AxP-GDL.git
```
The directory structure of the framework is as follows:
```
esm-AxP-GDL
├── best_models                                 <- Top models created using this framework. 
│   ├── amp_top1_esmt36_d10_hd256               <- Best model.           
│   │   ├── Metrics.txt                         <- Matthew correlation coefficient (MCC) achieved by this model. 
│   │   ├── Parameters.txt                      <- Parameters used to build the model.
│   ├── amp_top2_esmt36_d10_hd128               <- Second-best model           
│   │   ├── Metrics.txt                         <- Matthew correlation coefficient (MCC) achieved by this model. 
│   │   ├── Parameters.txt                      <- Parameters used to build the model.
├── datasets                                    <- Input comma separated value (CSV) file.
│   ├── AMPDiscover                                  
│   │   ├── AMPDiscover.csv                     <- Dataset used to evaluate the usefulness of the proposed framework.              
│   │   ├── Test(reduced-30).csv                <- Reduced test set built from AMPDiscover test set and comprised of sequences of up to 30 amino acids.                
│   │   ├── Test(reduced-100).csv               <- Reduced test set built from AMPDiscover test set and comprised of sequences of up to 100 amino acids.
├── graph                                       <- Scripts to build graphs.
│   ├── construct_graphs.py                     
│   ├── residues_level_features_encoding.py     
│   ├── structure_feature_extraction.py         
├── misc                                        <- Additional library to be installed to use the framework. 
│   ├──linux-64_pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2
├── models                                      <- Models used by this framework.
│   ├── esm2                                    <- ESM-2 module.
│   │   ├── checkpoints                         <- Directory where the ESM-2 models are downloaded.
│   │   ├── _init_.py                           <- Makes ESM-2 a Python module.
│   │   ├── esm2_representation.py              <- Script to use the ESM-2 models.
│   │   ├── esm2_representation.json            <- JSON with the ESM-2 representations that can be used.
│   ├── esmfold                                 <- ESMFold module.
│   │   ├── checkpoints                         <- Directory where the ESMFold model is downloaded.
│   │   ├── _init_.py                           <- Makes ESMFold a Python module.
│   │   ├── esmfold.py                          <- Script to use the ESMFold model.
│   ├── GAT                                     <- GAT module.
│   │   ├── _init_.py                           <- Makes GAT a Python module.
│   │   ├── GAT.py                              <- Script to use the Graph Attention Network (GAT) architecture.
│   ├── _init_.py                               
├── tool                                         
│   ├── data_preprocessing                       
│   │   ├── _init_.py                           
│   │   ├── data_preprocessing.py               <- Script to load and validate the input datasets.
│   ├── _init_.py                               
├── docker-compose.yml                          <- Configuration of the Docker container required by the framework.
├── Dockerfile                                  <- Docker image with all the dependencies required by the framework. 
├── README.md                                   <- README to use this framework
├── requirements.txt                            <- Python libraries used in this project.
├── test.py                                     <- Script to use a model for inference.
├── test.sh                                     <- Example to use a model for inference.
├── train.py                                    <- Script to train a model.
├── train.sh                                    <- Example script for training.
```

## **Dependencies**
This framework is currently supported for Linux, Python 3.7, CUDA 11 and Pytorch 1.12. The major dependencies used in this project are as follows:

>C++ compiler: https://gcc.gnu.org/ </br>
CUDA Toolkit: https://developer.nvidia.com/ </br>
Python: 3.7 </br>
PyTorch: 1.12.0+cu113 </br>
PyTorch Geometric: (torch-cluster: 1.6.1, torch-scatter: 2.0.9, torch-sparse: 0.6.15, torch-geometric: 2.3.1) </br>
ESM-2 (fair-esm:2.0.0) </br> 
ESMFold (fair-esm:2.0.0) 

Additional libraries used in this project are specified in `requirements.txt`. 

### **Installation (Linux)**
> **Pytorch**, **PyTorch Geometric**, **ESM-2**, **ESMFold** and the dependencies into requirements.txt are installed as follows:
#### PyTorch:
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```
#### PyTorch Geometric:
###### torch-cluster, torch-sparse and torch-geometric:
```
pip install --no-cache-dir torch-sparse==0.6.15 torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
```
###### torch-scatter:
```
conda install -y /misc/linux-64_pytorch-scatter-2.0.9-py37_torch_1.12.0_cu113.tar.bz2
```
#### ESM-2:
```
pip install fair-esm
```
#### ESMFold:
```
pip install --no-cache-dir fair-esm[esmfold]
pip install --no-cache-dir 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install --no-cache-dir 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```   
#### requirements.txt:
```
pip install --no-cache-dir requirements.txt
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
usage: train.py [-h] --dataset DATASET
                [--esm2_representation {esm2_t6,esm2_t12,esm2_t30,esm2_t36,esm2_t48}]
                [--tertiary_structure_method {esmfold}]
                --tertiary_structure_path TERTIARY_STRUCTURE_PATH
                [--tertiary_structure_load_pdbs] [--lr LR] [--drop DROP]
                [--e E] [--b B] [--hd HD]
                [--pretrained_model PRETRAINED_MODEL] --path_to_save_models
                PATH_TO_SAVE_MODELS [--heads HEADS] [--d D]
                [--log_file_name LOG_FILE_NAME]

optional arguments: 
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the input dataset in csv format
  --esm2_representation {esm2_t6,esm2_t12,esm2_t30,esm2_t36,esm2_t48}
                        ESM-2 model to be used
  --tertiary_structure_method {esmfold}
                        3D structure prediction method
  --tertiary_structure_path TERTIARY_STRUCTURE_PATH
                        Path to load or save the generated tertiary structures
  --tertiary_structure_load_pdbs
                        True if specified, otherwise, False. True indicates to
                        load existing tertiary structures from PDB files.
  --lr LR               Learning rate
  --drop DROP           Dropout rate
  --e E                 Maximum number of epochs
  --b B                 Batch size
  --hd HD               Hidden layer dimension
  --path_to_save_models PATH_TO_SAVE_MODELS
                        The path to save the trained models
  --heads HEADS         Number of heads
  --d D                 Distance threshold to construct graph edges
  --log_file_name LOG_FILE_NAME
                        Log file name                
```

#### Test
```
usage: test.py [-h] --dataset DATASET
               [--esm2_representation  {esm2_t6,esm2_t12,esm2_t30,esm2_t36,esm2_t48}]
               [--tertiary_structure_method {esmfold}]
               --tertiary_structure_path TERTIARY_STRUCTURE_PATH
               [--tertiary_structure_load_pdbs] --trained_model_path
               TRAINED_MODEL_PATH [--b B] [--drop DROP] [--hd HD]
               [--heads HEADS] [--d D] [--log_file_name LOG_FILE_NAME]
               [--test_result_file_name TEST_RESULT_FILE_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the input dataset in csv format
  --esm2_representation {esm2_t6,esm2_t12,esm2_t30,esm2_t36,esm2_t48}
                        ESM-2 model to be used
  --tertiary_structure_method {esmfold}
                        3D structure prediction method
  --tertiary_structure_path TERTIARY_STRUCTURE_PATH
                        Path to load or save the generated tertiary structures
  --tertiary_structure_load_pdbs
                        True if specified, otherwise, False. True indicates to
                        load existing tertiary structures from PDB files.
  --trained_model_path TRAINED_MODEL_PATH
                        The directory where the trained model to be used for inference is saved
  --b B                 Batch size
  --drop DROP           Dropout rate
  --hd HD               Hidden layer dimension
  --heads HEADS         Number of heads
  --d D                 Distance threshold to construct graph edges
  --log_file_name LOG_FILE_NAME
                        Log file name
  --test_result_file_name TEST_RESULT_FILE_NAME
                        Results file                        
```
### **Example**
We provide the train.sh and test.sh example scripts to train or use a model for inference, respectively.
In these scripts are used the AMPDiscover dataset as input set, the model `esm2_t36_3B_UR50D` to evolutionary 
characterize the graph nodes, a `distance threshold equal to 10 angstroms`
to build the graph edges, and a `hidden layer size equal to 128`.

When using the Docker container the example scripts should be used as follows:
```
docker-compose run --rm esm-axp-gdl-env sh train.sh
```
```
docker-compose run --rm esm-axp-gdl-env sh test.sh
```

### **Best models**
Top two models created with AMPDiscover using the esm-AxP-GDL framework are as follows:  

| Name                                                             | Dataset                                                          | Endpoint     | MCC    | Description                                                                                                                                                                                                                                                    |
|------------------------------------------------------------------|------------------------------------------------------------------|--------------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [amp_esmt36_d10_hd256.pt](https://drive.google.com/file/d/1kRbinBqNhGclUtf4VSAL4Hz7O0lJhjCO/view?usp=sharing) | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9521 | This model was created using the AMPDiscover dataset as input data, the model `esm2_t36_3B_UR50D` to evolutionary characterize the graph nodes, a `distance threshold equal to 10 angstroms` to build the graph edges, and a `hidden layer size equal to 256`. |
| [amp_esmt36_d10_hd128.pt](https://drive.google.com/file/d/1kWkRLIZ4KErcSfueQbxcJB3NtoUuXdae/view?usp=sharing) | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9505 | This model was created using the AMPDiscover dataset as input data, the model `esm2_t36_3B_UR50D` to evolutionary characterize the graph nodes, a `distance threshold equal to 10 angstroms` to build the graph edges, and a `hidden layer size equal to 128`. |

NOTE:  The performance `metrics` obtained and `parameters` used to build the best models are available at `/best_models` directory. The models are available-freely making click on the Table.