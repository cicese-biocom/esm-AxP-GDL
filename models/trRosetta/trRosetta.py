import tools.hhblits.hhblits_search as hh
import models.trRosetta.predict_many as tr
import argparse
import os
from tools.data_preprocessing.dataset_processing import load_and_validate_dataset


def trRosetta(args):
    """
    """
    try:
        abs_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        dataset = args.dataset

        dataset_path = os.path.join(abs_path, dataset)
        output_path = os.path.join(abs_path, args.output_path)

        # model, script and database required by trRosetta and hhblits
        hhb_database = os.path.join(abs_path, 'tools/hhblits/uniclust30_2018_08/uniclust30_2018_08')
        rosetta_model = os.path.join(abs_path, 'models/trRosetta/model2019_07')

        # Load and validation data_preprocessing dataset
        data = load_and_validate_dataset(dataset_path)

        # Create the 'a3m' folder
        a3m_path = os.path.join(output_path, 'a3m/')
        os.makedirs(a3m_path, exist_ok=True)

        # Create the 'hhm' folder
        hhm_path = os.path.join(output_path, 'hhm/')
        os.makedirs(hhm_path, exist_ok=True)

        # Create the 'tmp' folder
        tmp_path = os.path.join(output_path, 'tmp/')
        os.makedirs(tmp_path, exist_ok=True)

        # Create the 'npz' folder
        npz_path = os.path.join(output_path, 'npz/')
        os.makedirs(npz_path, exist_ok=True)

        # Run hhblits
        hh.run('hhblits', data, a3m_path, hhm_path, tmp_path, hhb_database)

        # Run trRoseTTA
        tr.predict(a3m_path, npz_path, rosetta_model)

    except Exception as e:
        raise

if __name__ == '__main__':
    # generate contact map, esm2 features before train and test model.
    parser = argparse.ArgumentParser()

    # Parameters
    parser.add_argument('-dataset', type=str, default='datasets/DeepAVPpred/DeepAVPpred.csv',
                        help='Input a file with data_preprocessing format with columns id, sequence, activity, partition')
    parser.add_argument('-output_path', type=str, default='datasets/DeepAVPpred/trRosetta_output/',
                        help='Folder path to save the output of trRoseTTA')

    args = parser.parse_args()
    trRosetta(args)