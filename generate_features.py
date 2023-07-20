import os
import utils.hhblits_search as hh
import yaml
import argparse

# Load the paths of tools and models
with open("config.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

# Tools
hhblits = cfg['hhblits']
rosetta = cfg['rosetta']

# Database
uniclust = cfg['uniclust']

# Models
rosetta_model = cfg['rosetta_model']


def generate_features(args):
    """
    """
    feas = args.feas
    if 'HHM' in feas:
        hh.run(hhblits, args.hhm_ifasta, args.hhm_oa3m, args.hhm_ohhm, args.hhm_tmp, uniclust)

    if 'NPZ' in feas:
        rosetta_cmd = 'python ' + rosetta + ' ' + \
                      args.tr_ia3m + ' ' + args.tr_onpz + ' -m ' + rosetta_model
        os.system(rosetta_cmd)


if __name__ == '__main__':      # Si generate_features.py se ejecuta directamente como un script
    # generate contact map, esm2 features before train and test model.
    parser = argparse.ArgumentParser()

    parser.add_argument('-feas', type=str, nargs='+', default=['HHM', 'NPZ'], help='Feature names')

    # HHblits parameters
    parser.add_argument('-hhm_ifasta', type=str, default='example2/test.fasta',
                        help='Input a file with fasta format for hhblits search')
    parser.add_argument('-hhm_oa3m', type=str, default='example2/a3m/',
                        help='Output folder saving .a3m files')

    # trRosetta parameters
    parser.add_argument('-tr_ia3m', type=str, default='example2/a3m/',
                        help='Input folder saving .a3m files')
    parser.add_argument('-tr_onpz', type=str, default='example2/npz/',
                        help='Output folder saving .npz files')
    parser.add_argument('-hhm_ohhm', type=str, default='example2/hhm/',
                        help='Output folder saving .hhm files')
    parser.add_argument('-hhm_tmp', type=str, default='example2/tmp/', help='Temp folder')

    args = parser.parse_args()

    generate_features(args)