import os
import yaml
import argparse

# Load the paths of tools and models
with open("config.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Tools
rosetta = cfg['rosetta']

# Models
rosetta_model = cfg['rosetta_model']


def generate_features(args):
    """
    """
    feas = args.feas

    if 'NPZ' in feas:
        rosetta_cmd = 'python ' + rosetta + ' ' + \
                      args.tr_ia3m + ' ' + args.tr_onpz + ' -m ' + rosetta_model
        os.system(rosetta_cmd)


if __name__ == '__main__':      # Si generate_features.py se ejecuta directamente como un script
    # generate contact map, esm2 features before train and test model.
    parser = argparse.ArgumentParser()

    parser.add_argument('-feas', type=str, nargs='+', default=['NPZ'], help='Feature names')

    # trRosetta parameters
    parser.add_argument('-tr_ia3m', type=str, default='example/a3m/',
                        help='Input folder saving .a3m files')
    parser.add_argument('-tr_onpz', type=str, default='example/npz/',
                        help='Output folder saving .npz files')

    args = parser.parse_args()

    generate_features(args)
