import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

from utils.json_parser import load_json
from workflow.ad_methods_collection_loader import ADMethodCollectionLoader


def set_of_functions(value):
    functions = value.replace(" ", "").split(',')
    functions = set(functions)
    return functions


def methods_for_ad():
    available_methods = ADMethodCollectionLoader().get_method_names()
    return ", ".join(available_methods)


class ArgsParserHandler:
    def __init__(self):
        self.parser = ArgumentParser()
        check_json_params_arg()

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

    def _add_eval_arguments(self) -> Dict:
        self._add_common_arguments()

        self.parser.add_argument('--output_path', type=Path, required=True,
                                 help='The path where the output data will be saved.')

        self.parser.add_argument('--seed', type=int, default=None,
                                 help=' Seed to run the Test/Inference mode')

        self.parser.add_argument('--methods_for_ad', type=set_of_functions, default=None,
                                 help=f"Methods to build applicability domain model. The options available are: "
                                      f"{methods_for_ad()}")

        self.parser.add_argument('--feature_file_for_ad', type=Path, default=None,
                                 help='Path of the CSV file of features to build the applicability domain.')

    def get_training_arguments(self) -> Dict:
        self._add_common_arguments()

        # See 10.1002/pro.4928 for reduced and combined representations
        self.parser.add_argument('--esm2_representation', type=str, default='esm2_t33',
                                 choices=['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36', 'esm2_t48',
                                          'reduced_esm2_t6', 'reduced_esm2_t12', 'reduced_esm2_t30', 'reduced_esm2_t33',
                                          'reduced_esm2_t36', 'combined_esm2'],
                                 help='ESM-2 representation to be used')

        self.parser.add_argument('--edge_construction_functions', type=set_of_functions, default=None,
                                 help="Criteria (e.g., distance) to define a relationship (graph edges) between amino "
                                      "acids. The options available are: 'distance_based_threshold', "
                                      "'sequence_based', 'esm2_contact_map'")

        self.parser.add_argument('--distance_function', type=str, default=None,
                                 choices=['euclidean', 'canberra', 'lance_williams', 'clark', 'soergel',
                                          'bhattacharyya',
                                          'angular_separation'],
                                 help='Distance function to construct the edges of the distance-based graph')

        self.parser.add_argument('--distance_threshold', type=float, default=None,
                                 help='Distance threshold to construct the edges of the distance-based graph')

        self.parser.add_argument('--esm2_model_for_contact_map', type=str, default=None,
                                 choices=['esm2_t6', 'esm2_t12', 'esm2_t30', 'esm2_t33', 'esm2_t36', 'esm2_t48'],
                                 help='ESM-2 model to be used to obtain ESM-2 contact map')

        self.parser.add_argument('--probability_threshold', type=float, default=None,
                                 help='Probability threshold for constructing a graph based on ESM-2 contact maps. '
                                      'It takes a value between 0.5 and 1.0.')

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

        self.parser.add_argument('--split_method', type=str, default=None,
                                 choices=['random', 'expectation_maximization'],
                                 help='Method to split an input dataset in training and validation sets. This parameter is used when an used-defined validation set is not given. To use this parameter, all no-test instances must be marked as training, i.e., value 1 in the input CSV file.')

        self.parser.add_argument('--split_training_fraction', type=float, default=None,
                                 help='If the --split_method is specified, this parameter represents the percentage of instances '
                                      'to be considered as training. The other ones will be allocated in the validation set. '
                                      'It takes a value between 0.6 and 0.9.')

        args = self.parser.parse_args()
        return vars(args)

    def get_eval_arguments(self) -> Dict:
        self._add_eval_arguments()

        args = self.parser.parse_args()
        return vars(args)

    def get_inference_arguments(self) -> Dict:
        self._add_eval_arguments()

        self.parser.add_argument('--inference_batch_size', type=int, default=20000,
                                 help='As the inference data are unlimited, this parameters contains the number of instances to be processed in a specific chunk (batch) at a time.')

        args = self.parser.parse_args()
        return vars(args)


def check_json_params_arg():
    user_args = sys.argv[1:]

    if '--json_params' in user_args:
        if len(user_args) > 2:
            raise ValueError("--json_params cannot be used together with any other parameters. "
                             "Please either provide this path or specify parameters individually, not both.")

        json_file_path = Path(user_args[1]).resolve()
        json_args = load_json(json_file_path)

        argv = []
        for key, value in json_args.items():
            if isinstance(value, bool):
                if value:
                    argv.append(f"--{key}")
            elif isinstance(value, list):
                if value:  # evitar listas vacías
                    comma_separated = ",".join(map(str, value))
                    argv.extend([f"--{key}", comma_separated])
            elif value is not None:
                argv.extend([f"--{key}", str(value)])

        sys.argv = [sys.argv[0]] + argv

