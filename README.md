[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/docs/1.12/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.3.1-%237732a8.svg?style=flat)](https://pytorch-geometric.readthedocs.io/en/2.3.1/)
[![CUDA](https://img.shields.io/badge/CUDA-11-%2376B900.svg?style=flat&logo=NVIDIA&logoColor=white)](https://developer.nvidia.com/cuda)
[![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

# esm-AxP-GDL

**esm-AxP-GDL** is a flexible **Graph Deep Learning (GDL)** framework designed to leverage graph-based representations derived from evolutionary-scale protein language models, namely **ESM-2** and **ESMFold**, for the modeling of peptide/protein activities.

The framework supports both:

- **Structure-dependent graphs**, derived from predicted tertiary structures using multiple distance functions, enabling the generation of diverse graph topologies that capture different structural relationships between amino acids.  
- **Structure-free graphs**, constructed directly from ESM-2 predicted contact maps, allowing structural information to be exploited without explicit structure prediction and thus improving computational efficiency.  

Graph node features are represented using **ESM-2 embeddings**, capturing rich evolutionary information.

esm-AxP-GDL supports both **classification and regression tasks**, and integrates multiple validation strategies, including embedding randomization, 
geometric coordinate perturbation, and random graph generation.  

Additionally, the framework enables comprehensive **applicability domain analysis** through one-class classification methods.

Overall, esm-AxP-GDL provides a flexible and reproducible platform that combines **Graph Deep Learning** with established **QSAR best practices** for peptide and protein activity modeling.

---

## **Install esm-AxP-GDL**
Clone the repository:
```
git clone https://github.com/cicese-biocom/esm-AxP-GDL.git
```
The directory structure of the framework is as follows:

```
esm-AxP-GDL
│
├── datasets/                                         <- Benchmark datasets used in experiments.
│   ├── AMPDiscover/
│   │   ├── AMPDiscover(Training-Validation-Test).csv
│   │   ├── Test(reduced-100).csv
│   │   ├── Test(reduced-30).csv
│   │   ├── AMP_External.csv
│   ├── AVPDiscover/
│   │   ├── AVPDiscover(Training-Validation-Test).csv
│   │   ├── AMP_External.csv
│
├── example/                                          <- Example datasets for quick execution.
│   ├── ExampleDataset.csv
│   ├── ExampleDatasetInference.csv
│
├── best_models/                                      <- Pretrained models and their configurations.
│   ├── AMPDiscover/                                  <- Models trained on AMPDiscover dataset.
│   │   ├── amp_esmt36_d10_hd128_(Model3)/
│   │   │   ├── Metrics.txt                           <- Model performance (e.g., MCC).
│   │   │   ├── Parameters.json                       <- Training configuration.
│   │   ├── amp_esmt33_d10_hd128_(Model2)/
│   │   ├── amp_esmt30_d15_hd128_(Model5)/
│
├── src/                                              <- Core framework source code.
│
│   ├── config/                                       <- Configuration definitions for core framework components.
│   │   ├── ad_methods_config.py
│   │   ├── amino_acid_descriptors_config.py
│   │   ├── esm2_representations_config.py
│   │   ├── features_config.py
│   │   ├── log_config.py
│   │   ├── outputs_config.py
│
│   ├── params/                                       <- Parameter definitions and validation for execution modes.
│   │   ├── common.py
│   │   ├── execution.py
│   │   ├── inference.py
│   │   ├── prediction.py
│   │   ├── test.py
│   │   ├── training.py
│
│   ├── utils/                                        <- Utility functions.
│   │   ├── base_parameters.py
│   │   ├── json.py
│   │   ├── path.py
│   │   ├── pdb.py
│
│   ├── data_processing/                              <- Data loading, partitioning, and preprocessing.
│   │   ├── data_loader.py
│   │   ├── data_partitioner.py
│   │   ├── data_processor.py
│   │   ├── target_feature_validator.py
│
│   ├── feature_extraction/                           <- Methods for extracting sequence features and graph representations of sequences, used for applicability domain and data partitioning through clustering-based strategy.
│   │   ├── config_loader.py                          
│   │   ├── methods.py                                
│
│   ├── models/                                       <- Pretrained models used to generate node features and structural information for graph construction.
│   │   ├── esm2.py
│   │   ├── esmfold.py
│
│   ├── graph_builder/                                <- Graph construction using multiple edge-building functions.
│   │   ├── distance_functions.py
│   │   ├── edge_build_functions.py
│   │   ├── graph_builder.py
│   │   ├── node_feature_builder.py
│   │   ├── tertiary_structures.py                    <- Structure prediction using ESMFold or loading from PDB files.
│
│   ├── applicability_domain/                         <- Methods for computing the Applicability Domain (AD).
│   │   ├── config_loader.py                          
│   │   ├── methods.py                                
│
│   ├── architectures/                                <- GNN model architectures.
│   │   ├── gnn.py
│   │   ├── gat_v1.py
│   │   ├── gat_v2.py
│
│   ├── modeling/                                     <- Model training, evaluation, and prediction.
│   │   ├── executor.py
│   │   ├── metrics.py
│   │   ├── model_selector.py
│   │   ├── output_processor.py
│   │   ├── prediction_maker.py
│   │   ├── prediction_stats.py
│
│   ├── workflow/                                     <- Coordinates the execution of the framework.
│   │   ├── application_context.py                    <- Manages and injects the dependencies required to run the framework.
│   │   ├── execution_factories.py                    <- Creates and prepares the tasks for training, test, or inference based on the configuration.
│   │   ├── gdl_workflow.py                           <- Defines the workflow steps according to the selected execution mode.
│   │   ├── logging_config.py                         <- Sets up how logs are generated and stored.
│
├── train.py                                          <- Training entry point.
├── test.py                                           <- Testing entry point.
├── inference.py                                      <- Inference entry point.
│
├── *.sh / *_SLURM.sh                                 <- Execution scripts (local & HPC).
│
├── .env                                               <- Environment variables (paths to configuration files).
├── environment.yml                                    <- Python libraries required.
├── Dockerfile                                         <- Docker image.
├── docker-compose.yml                                 <- Container configuration.
│
├── README.md
```
---

## **Dependencies**
This framework is currently supported for Linux, Python 3.9, CUDA 11 and Pytorch 1.12.0. The major dependencies used in this project are:

- **Python**: 3.9  
- **CUDA Toolkit**: 11  
- **PyTorch**: 1.12.0+cu113  
- **PyTorch Geometric**: 2.3.1  
  - torch-cluster: 1.6.0  
  - torch-scatter: 2.1.0  
  - torch-sparse: 0.6.15  
  - spline-conv: 1.2.1  
- **ESM-2 / ESMFold**: fair-esm 2.0.0  
- **C++ compiler**: https://gcc.gnu.org/  
- **Java 11** (required for Expectation-Maximization data partitioning via `python-weka-wrapper3`)

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
1. module purge
2. module load python/ondemand-jupyter-python3.8
3. module load gcc/9.2.0
4. module load cuda/11.3.0
5. module load java/1.8.0_181-oracle
6. conda env create -f environment.yml
```

NOTE: we provide template scripts to run training/test/inference Slurm batch jobs.

### Environment Variables

The framework relies on a `.env` file to define the paths to configuration files required during execution.

```
LOG_CONFIG_FILE=src/config/log_config.json
ESM_CHECKPOINTS_DIR=src/models
ESM2_REPRESENTATION_CONFIG_FILE=src/config/esm2_representations_config.json
AMINO_ACID_DESCRIPTORS_FILE=src/config/amino_acid_descriptors_config.csv
FEATURES_CONFIG_FILE=src/config/features_config.json
AD_METHODS_CONFIG_FILE=src/config/ad_methods_config.json
OUTPUTS_CONFIG=src/config/outputs_config.json
```
---

## **Usage**
### **Input data format**
The esm-AxP-GDL framework is inputted with a comma-separated value (CSV) file as input which contains the identifier, the amino acid sequence, the activity, and the partition to which each peptide/protein 
belongs (i.e., training, validation, and test). The training, validation, and test partitions are denoted by the numbers 1, 2, and 3, respectively. If a validation partition is not specified, then the 
input training partition is divided into training and validation sets following a random or clustering-based strategy.

As for the activity column in the CSV files, it can consist of continuous or discrete values for modeling regression or classification tasks, respectively. The discrete values must be consecutive 
integers starting from 0, for example, 0 (false) and 1 (true) for binary classification tasks; 0, 1, and 2 for ternary classification tasks; and so on for multi-class classification tasks.

To run the inference mode, a FASTA file or a CSV file can be specified as input. In this case, the CSV file will only have two columns: one for the identifier and another for the peptide/protein sequences 
to be screened.

### **For training or using a model for testing or inference**
The scripts `train.py`, `test.py`, and `inference.py` are used to perform the training, test, and inference stages, respectively. 
The following command-line examples illustrate how to execute each step.

#### Train
```
usage: train.py [-h] [--dataset DATASET] [--load-tertiary-structure] [--pdb-path PDB_PATH] [--batch-size BATCH_SIZE] [--gdl-model-path GDL_MODEL_PATH] 
                [--command-line-build-graphs-parameters COMMAND_LINE_BUILD_GRAPHS_PARAMETERS][--esm2-representation {esm2_t6, esm2_t12, esm2_t30, esm2_t33, esm2_t36, esm2_t48, reduced_esm2_t6, reduced_esm2_t12, reduced_esm2_t30, reduced_esm2_t33, reduced_esm2_t36, combined_ESM2}]
                [--edge-build-functions EDGE_BUILD_FUNCTIONS [distance_based_threshold, sequence_based, esm2_contact_map, empty_graph]] 
                [--distance-function {euclidean, canberra, lance_williams, clark, soergel, bhattacharyya, cosine}]
                [--distance-threshold DISTANCE_THRESHOLD] [--esm2-model-for-contact-map {esm2_t6, esm2_t12, esm2_t30, esm2_t33, esm2_t36, esm2_t48}] [--probability-threshold PROBABILITY_THRESHOLD]
                [--number-of-heads NUMBER_OF_HEADS] [--hidden-layer-dimension HIDDEN_LAYER_DIMENSION] [--add-self-loops] [--use-edge-attr] [--learning-rate LEARNING_RATE] [--dropout-rate DROPOUT_RATE]
                [--pooling-ratio POOLING_RATIO] [--number-of-epochs NUMBER_OF_EPOCHS] [--save-ckpt-per-epoch] [--validation-method {random_coordinates, random_embeddings, random_graphs}]
                [--randomness-percentage RANDOMNESS_PERCENTAGE] [--probability-for-edge-creation PROBABILITY_FOR_EDGE_CREATION] [--seed-for-edge-creation SEED_FOR_EDGE_CREATION]
                [--split-method {random, expectation_maximization}] [--split-training-fraction SPLIT_TRAINING_FRACTION] [--gdl-architecture {GATv1, GATv2}]
                [--modeling-task {binary_classification, multiclass_classification, regression}] [--numbers-of-class NUMBERS_OF_CLASS]

required arguments:
  --modeling-task {binary_classification, multiclass_classification, regression}
                        Type of modeling task to execute
  --dataset DATASET     Path to the input dataset (CSV for training/test, CSV or FASTA for inference)
  --gdl-model-path GDL_MODEL_PATH
                        Path to trained models or for loading a trained model in test/inference mode

optional arguments:
  --command-line-build-graphs-parameters COMMAND_LINE_BUILD_GRAPHS_PARAMETERS
                        Path to a JSON file with command line parameters (default: None)
  --numbers-of-class NUMBERS_OF_CLASS
                        Number of classes to predict (required if modeling_task is 'multiclass'). (default: None)                        
  --esm2-representation {esm2_t6, esm2_t12, esm2_t30, esm2_t33, esm2_t36, esm2_t48, reduced_esm2_t6, reduced_esm2_t12, reduced_esm2_t30, reduced_esm2_t33, reduced_esm2_t36, combined_ESM2}
                        ESM-2 representation to be used (default: esm2_t33)
  --edge-build-functions EDGE_BUILD_FUNCTIONS [distance_based_threshold, sequence_based, esm2_contact_map, empty_graph]
                        Functions to build edges (default: None).
  --distance-function {euclidean, canberra, lance_williams, clark, soergel, bhattacharyya, cosine}
                        Distance function to construct the edges of the distance-based graph (default: None)
  --distance-threshold DISTANCE_THRESHOLD
                        Distance threshold to construct the edges of the distance-based graph (default: None)
  --esm2-model-for-contact-map {esm2_t6, esm2_t12, esm2_t30, esm2_t33, esm2_t36, esm2_t48}
                        ESM-2 model to be used to obtain ESM-2 contact map (default: None)
  --probability-threshold PROBABILITY_THRESHOLD
                        Probability threshold for constructing a graph based on ESM-2 contact maps (default: None)                                                                                       
  --load-tertiary-structure
                        If True, load tertiary structures; otherwise predict tertiary structures with ESMFold (default: None)
  --pdb-path PDB_PATH   Path where tertiary structures are saved or loaded from PDB files (default: None)
  --split-method {random, expectation_maximization}
                        Method to split an input dataset in training and validation sets. This parameter is used when an used-defined validation set is not given. To use this parameter, all no-test instances
                        must be marked as training, i.e., value 1 in the input CSV file. (default: None)
  --split-training-fraction SPLIT_TRAINING_FRACTION
                        If the --split_method is specified, this parameter represents the percentage of instances to be considered as training. The other ones will be allocated in the validation set. It
                        takes a value between 0.6 and 0.9. (default: None)
  --gdl-architecture {GATv1, GATv2}
                        GDL architectures to use (default: GATv1)
  --hidden-layer-dimension HIDDEN_LAYER_DIMENSION
                        Hidden layer dimension (default: 128)
  --learning-rate LEARNING_RATE
                        Learning rate (default: 0.0001)
  --dropout-rate DROPOUT_RATE
                        Dropout rate (default: 0.25)
  --pooling-ratio POOLING_RATIO
                        Pooling ratio (default: 10)
  --add-self-loops      True if specified, otherwise, False. True indicates to use auto loops in attention layer (default: False)
  --use-edge-attr       True if specified, otherwise, False. True indicates to use edge attributes in graph learning (default: False)                        
  --number-of-epochs NUMBER_OF_EPOCHS
                        Maximum number of epochs (default: 200)
  --number-of-heads NUMBER_OF_HEADS
                        Number of heads (default: 8)         
  --batch-size BATCH_SIZE
                        Batch size (default: 512)                                       
  --save-ckpt-per-epoch
                        True if specified, otherwise, False. True indicates that the models of every epoch will be saved. False indicates that the latest model and the best model regarding the MCC metric
                        will be saved (default: False)
  --validation-method {random_coordinates, random_embeddings, random_graphs}
                        Validation strategy to assess whether model predictions are not obtained by chance. random_coordinates randomizes node geometric coordinates to evaluate structural dependence; 
                        random_embeddings shuffles node features to assess feature importance; random_graphs uses Erdős–Rényi model graphs as a baseline to evaluate graph structure relevance. 
                        If not specified, no validation is applied. (default: None)
  --randomness-percentage RANDOMNESS_PERCENTAGE
                        Percentage of nodes to be perturbed during validation. For 'random_embeddings', it defines the fraction of node features to shuffle; for 'random_coordinates', the fraction of node
                        geometric coordinates to randomize. (default: None)
  --probability-for-edge-creation PROBABILITY_FOR_EDGE_CREATION
                        Probability of edge creation in the Erdős-Rényi model used in 'random_graphs'. Controls the density of the generated random graph. (default: None)
  --seed-for-edge-creation SEED_FOR_EDGE_CREATION
                        Random seed used for reproducible generation of edges in the Erdős-Rényi graph for 'random_graphs' validation. (default: None)

help:
  -h, --help            Show this help message and exit
```

#### Test
```
usage: test.py [-h] [--dataset DATASET] [--load-tertiary-structure] [--pdb-path PDB_PATH] [--batch-size BATCH_SIZE] [--gdl-model-path GDL_MODEL_PATH]
               [--command-line-build-graphs-parameters COMMAND_LINE_BUILD_GRAPHS_PARAMETERS] [--output-path OUTPUT_PATH] [--seed SEED] [--calculate-ad] 
               [--methods-for-ad METHODS_FOR_AD [percentile_based(gc), percentile_based(perp), IF(perp_aad), IF(gc), IF(gc_perp_aad), IF(aad)]]
               [--feature-file-for-ad FEATURE_FILE_FOR_AD] [--prediction-batch-size PREDICTION_BATCH_SIZE]

required arguments:
  --dataset DATASET     Path to the input dataset (CSV for training/test, CSV or FASTA for inference)
  --gdl-model-path GDL_MODEL_PATH
                        Path to trained models or for loading a trained model in test/inference mode
  --output-path OUTPUT_PATH
                        The path where the output data will be saved

optional arguments:
  --command-line-build-graphs-parameters COMMAND_LINE_BUILD_GRAPHS_PARAMETERS
                        Path to a JSON file with command line parameters (default: None)
  --seed SEED           User-defined random seed to ensure deterministic behavior. (default: None)                        
  --load-tertiary-structure
                        If True, load tertiary structures; otherwise predict tertiary structures with ESMFold (default: False)
  --pdb-path PDB_PATH   Path where tertiary structures are saved or loaded from PDB files (default: None)
  --batch-size BATCH_SIZE
                        Number of instances processed in each batch during model evaluation. (default: 512)
  --calculate-ad        True if specified, otherwise, False. True indicates to calculate applicability domain (default: False)
  --methods-for-ad METHODS_FOR_AD [percentile_based(gc), percentile_based(perp), IF(perp_aad), IF(gc), IF(gc_perp_aad), IF(aad)]
                        Methods to build applicability domain model (default: None)
  --feature-file-for-ad FEATURE_FILE_FOR_AD
                        Path of the CSV file of features to build the applicability domain (default: None)
  --prediction-batch-size PREDICTION_BATCH_SIZE
                        Number of instances to calculate in a single batch during prediction. (default: 20000)

help:
  -h, --help            show this help message and exit                                       
```
#### Inference
```
usage: inference.py [-h] [--dataset DATASET] [--load-tertiary-structure] [--pdb-path PDB_PATH] [--batch-size BATCH_SIZE] [--gdl-model-path GDL_MODEL_PATH]
                    [--command-line-build-graphs-parameters COMMAND_LINE_BUILD_GRAPHS_PARAMETERS] [--output-path OUTPUT_PATH] [--seed SEED] [--calculate-ad]
                    [--methods-for-ad METHODS_FOR_AD [percentile_based(gc), percentile_based(perp), IF(perp_aad), IF(gc), IF(gc_perp_aad), IF(aad)]]] 
                    [--feature-file-for-ad FEATURE_FILE_FOR_AD] [--prediction-batch-size PREDICTION_BATCH_SIZE]

required arguments:
  --dataset DATASET     Path to the input dataset (CSV for training/test, CSV or FASTA for inference)
  --gdl-model-path GDL_MODEL_PATH
                        Path to trained models or for loading a trained model in test/inference mode
  --output-path OUTPUT_PATH
                        The path where the output data will be saved

optional arguments:
  --command-line-build-graphs-parameters COMMAND_LINE_BUILD_GRAPHS_PARAMETERS
                        Path to a JSON file with command line parameters (default: None)
  --seed SEED           User-defined random seed to ensure deterministic behavior. (default: None)                        
  --load-tertiary-structure
                        If True, load tertiary structures; otherwise predict tertiary structures with ESMFold (default: False)
  --pdb-path PDB_PATH   Path where tertiary structures are saved or loaded from PDB files (default: None)
  --batch-size BATCH_SIZE
                        Batch size (default: 512)
  --calculate-ad        True if specified, otherwise, False. True indicates to calculate applicability domain (default: False)
  --methods-for-ad METHODS_FOR_AD [percentile_based(gc), percentile_based(perp), IF(perp_aad), IF(gc), IF(gc_perp_aad), IF(aad)]
                        Methods to build applicability domain model (default: None)
  --feature-file-for-ad FEATURE_FILE_FOR_AD
                        Path of the CSV file of features to build the applicability domain (default: None)
  --prediction-batch-size PREDICTION_BATCH_SIZE
                        Number of instances to calculate in a single batch during prediction. (default: 20000)

help:
  -h, --help            show this help message and exit                                         
```

---

### **Example**
We provide the train.sh and test.sh example scripts to train or use a model for inference, respectively.
In these scripts are used the AMPDiscover dataset as an input set, the model `esm2_t36_3B_UR50D` to evolutionarily 
characterize the graph nodes, a `distance threshold equal to 10 angstroms`
to build the graph edges, and a `hidden layer size equal to 128`.

When using the Docker container, the example scripts should be used as follows:
```
docker-compose run --rm esm-axp-gdl-env-py39 sh train.sh
```
```
docker-compose run --rm esm-axp-gdl-env-py39 sh test.sh
```
```
docker-compose run --rm esm-axp-gdl-env-py39 sh inference.sh
```
---

## Best Models

Best-performing models generated with this framework for **general antimicrobial peptide (general-AMP) prediction**. All models were trained using the **AMPDiscover** dataset and mainly differ in 
the ESM-2 representation and graph construction parameters. These configurations achieve state-of-the-art performance in terms of MCC.

| Name                                                                                                              | Dataset                                                          | Endpoint     | MCC    | Description |
|-------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|--------------|--------|------------|
| [amp_esmt33_d10_hd128 (Model2)](https://drive.google.com/uc?export=download&id=1edd1oa7YqQwXufGp432ApmZOv2ou3TmJ) | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9389 | Built using `esm2_t33_650M_UR50D` embeddings, a distance threshold of 10 Å for graph construction, and a hidden layer dimension of 128. |
| [amp_esmt36_d10_hd128 (Model3)](https://drive.google.com/uc?export=download&id=1Ba2hL2EqpMtM3Z8utm9DbwBczL6UX4TD) | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9505 | Built using `esm2_t36_3B_UR50D` embeddings, a distance threshold of 10 Å for graph construction, and a hidden layer dimension of 128. |
| [amp_esmt30_d15_hd128 (Model5)](https://drive.google.com/uc?export=download&id=1zmKru367VD798iIGgyHTWrbWaPUUFvjq) | [AMPDiscover](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251) | general-AMPs | 0.9379 | Built using `esm2_t30_150M_UR50D` embeddings, a distance threshold of 15 Å for graph construction, and a hidden layer dimension of 128. |

NOTE: The performance `metrics` obtained and `parameters` used to build the best models are available at `/best_models` directory. The models are available-freely making click on the Table.

---

## Related Publications

The development and application of **esm-AxP-GDL** are supported by the following papers:

- Cordoves-Delgado, G., & García-Jacas, C. R. (2024). Predicting antimicrobial peptides using ESMFold-predicted structures and ESM-2-based amino acid features with graph deep learning. *Journal of Chemical Information and Modeling, 64*(10), 4310–4321. https://doi.org/10.1021/acs.jcim.3c02061

- Cordoves-Delgado, G., García-Jacas, C. R., Marrero-Ponce, Y., Aguila, S. A., & Lizama-Uc, G. (2026). Leveraging different distance functions to predict antiviral peptides with geometric deep learning from ESMFold-predicted tertiary structures. *Antibiotics, 15*(1), 39. https://doi.org/10.3390/antibiotics15010039

