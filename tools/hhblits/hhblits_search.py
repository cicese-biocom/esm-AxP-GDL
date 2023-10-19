import os
import argparse
from tqdm.std import trange


def load_fasta(fasta_file):
    names = []
    seqs = []
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        print("Length:", len(lines) / 2)
        for line in lines:
            line = line.strip()
            if line[0] == '>':
                names.append(line)
            else:
                seqs.append(line)
    return names, seqs


def run(hhb, data, tg, tg_hhm, tmp_folder, db):
    """
    Using HHblits to search against Uniclust2018 database, to generate .a3m and .hhm files
    parameters:
        :param hhb: path of the hhblits
        :param data: identifiers and sequences itself. Containing multiple sequences
        :param tg: target output .a3m folder
        :param tg_hhm: target output .hhm folder
        :param tmp_folder: tmp folder saving the .fasta files containing a singe sequence that is split from the input files
        :param db: the path of Uniclust2018
    """
    names = data.id.apply(lambda x: '>' + x)
    seqs = data.sequence

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    for f in os.listdir(tmp_folder):
        os.remove(tmp_folder + f)

    for i in range(len(names)):
        name = names[i]
        fname = name.replace('|', '_')[1:]
        seq = seqs[i]
        with open(tmp_folder + fname + '.fasta', 'w') as f:
            f.write(name + '\n')
            f.write(seq)

    #tg_hhm = os.path.join(tg_hhm, 'output/')
    if not os.path.exists(tg):
        os.makedirs(tg)
 #   if not os.path.exists(tg_hhm):
 #       os.makedirs(tg_hhm)

    try:
        for i in trange(len(names)):
            name = names[i]
            fname = name.replace('|', '_')[1:]
            fn = tmp_folder + fname + '.fasta'
            cmd = hhb + \
                  ' -i ' + fn + \
                  ' -oa3m ' + tg + fname + '.a3m' \
                  ' -d ' + db + ' -cpu 8 -v 2 -n 3 -e 0.01'
            # os.system(cmd)

            print(f'cmd for {fname} generated')

            with open("log_hhblits.txt", "w") as log_file:
                os.system(cmd + " > log_hhblits.txt 2>&1")

            print(f'File {fname} .a3m generated')
    except:
        print('Failed to search !')
    finally:
        # remove temp folder
        for f in os.listdir(tmp_folder):
            os.remove(tmp_folder + f)


def main(args):
    names = args.data.id.apply(lambda x: '>' + x)
    seqs = args.data.sequence
    pt = args.pt
    db = args.d
    tg = args.oa3m
    tg_hhm = args.ohhm

    tmp_folder = './tmp/' + pt + '/'

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    for i in range(len(names)):
        name = names[i]
        fname = name.replace('|', '_')[1:]
        seq = seqs[i]
        with open(tmp_folder + fname + '.fasta', 'w') as f:
            f.write(name + '\n')
            f.write(seq)

    if not os.path.exists(tg + pt):
        os.makedirs(tg + pt)
    if not os.path.exists(tg + pt):
        os.makedirs(tg_hhm + pt)

    try:
        for i in trange(len(names)):
            name = names[i]
            fname = name.replace('|', '_')[1:]
            fn = tmp_folder + fname + '.fasta'
            cmd = 'hhblits -i ' + fn + ' -o ' + tg + 'tmp.hhr' + \
                  ' -oa3m ' + tg + pt + '/' + fname + '.a3m -ohhm ' + tg_hhm + pt + '/' + fname + '.hhm' + \
                  ' -d ' + db + ' -cpu 8 -v 0 -n 3 -e 0.01'
            os.system(cmd)
    except:
        print('Failed to search !')
    finally:
        # remove temp folder
        for f in os.listdir(tmp_folder):
            os.remove(tmp_folder + f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', type=str, default='AMP', help='Peptide type, used for name the output files')
    parser.add_argument('-d', type=str, help='Database that is searched against')
    parser.add_argument('-data', type=str, help='Sequences, identifiers and sequences itself')
    parser.add_argument('-oa3m', type=str, default='result_a3m/', help='Output folder saving o3m files')
    parser.add_argument('-ohhm', type=str, default='result_hhm/', help='Output folder saving hhm files')
    args = parser.parse_args()
    main(args)