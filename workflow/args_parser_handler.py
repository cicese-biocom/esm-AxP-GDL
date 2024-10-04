from argparse import ArgumentParser
from pathlib import Path
from typing import Dict


def set_of_edge_functions(value):
    functions = value.replace(" ", "").split(',')
    functions = set(functions)
    return functions


class ArgsParserHandler:
    def __init__(self):
        self.parser = ArgumentParser()

    def _add_common_arguments(self):
        self.parser.add_argument('--dataset', type=Path, required=True, help='Path to the input dataset in CSV format')

        self.parser.add_argument('--tertiary_structure_method', type=str, default=None,
                                 choices=['esmfold'],
                                 help='3D structure prediction method. None indicates to load existing tertiary '
                                      'structures from PDB files , otherwise, sequences in input CSV file are '
                                      'predicted using the specified method')

        self.parser.add_argument('--pdb_path', type=Path, default=None,
                                 help='Path where tertiary structures are saved in or loaded from PDB files')

        self.parser.add_argument('--batch_size', type=int, default=512, help='Batch size')

        self.parser.add_argument('--gdl_model_path', type=Path, required=True,
                                 help=' The path to save/load the models')

    def get_training_arguments(self) -> Dict:
        self._add_common_arguments()
        self.parser.add_argument('--esm2_representation', type=str, default='esm2_t33',
                                 choices=['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36', 'esm2_t48'],
                                 help='ESM-2 model to be used')

        self.parser.add_argument('--edge_construction_functions', type=set_of_edge_functions, default=None,
                                 help="Criteria (e.g., distance) to define a relationship (graph edges) between amino "
                                      "acids. Only one ESM-2 contact map can be specified. "
                                      "The options available are: 'distance_based_threshold', "
                                      "'sequence_based', 'esm2_contact_map_50', 'esm2_contact_map_60', "
                                      "'esm2_contact_map_70', 'esm2_contact_map_80', 'esm2_contact_map_90'")
                                 
        self.parser.add_argument('--distance_function', type=str, default=None,
                                 choices=['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel',
                                          'bhattacharyya',
                                          'angular_separation'],
                                 help='Distance function to construct graph edges')

        self.parser.add_argument('--distance_threshold', type=float, default=None,
                                 help='Distance threshold to construct graph edges')

        self.parser.add_argument('--amino_acid_representation', type=str, default='CA',
                                 choices=['CA'],
                                 help='Reference atom into an amino acid to define a relationship (e.g., distance) '
                                      'regarding another amino acid') 

        self.parser.add_argument('--number_of_heads', type=int, default=8, help='Number of heads')

        self.parser.add_argument('--hidden_layer_dimension', type=int, default=128, help='Hidden layer dimension')

        self.parser.add_argument('--add_self_loops', action="store_true",
                                 help="True if specified, otherwise, False. True indicates to use auto loops in "
                                      "attention layer.")

        self.parser.add_argument('--use_edge_attr', action="store_true",
                                 help="True if specified, otherwise, False. True indicates to use edge attributes in "
                                      "graph "
                                      "learning.")

        self.parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

        self.parser.add_argument('--dropout_rate', type=float, default=0.25, help='Dropout rate')

        self.parser.add_argument('--number_of_epochs', type=int, default=200, help='Maximum number of epochs')

        self.parser.add_argument('--save_ckpt_per_epoch', action="store_true",
                                 help="True if specified, otherwise, False. True indicates that the models of every "
                                      "epoch will be saved. False indicates that the latest model and the best model "
                                      "regarding the MCC metric will be saved.") 

        self.parser.add_argument('--validation_mode', type=str, default=None,
                                 choices=['random_coordinates', 'random_embeddings'],
                                 help='Criteria to validate that the predictions of the models are not by chance')

        self.parser.add_argument('--randomness_percentage', type=float, default=None,
                                 help='Percentage of rows to be randomly generated. This parameter and the --validation_mode parameter are used together.')

        args = self.parser.parse_args()
        return vars(args)

    def get_eval_arguments(self) -> Dict:
        self._add_common_arguments()

        self.parser.add_argument('--output_path', type=Path, required=True,
                                 help='The path where the output data will be saved.')   

        self.parser.add_argument('--seed', type=int, default=None,
                                 help=' Seed to run the Test/Inference mode')

        args = self.parser.parse_args()
        return vars(args)
