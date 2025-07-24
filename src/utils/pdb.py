import warnings
from pathlib import Path

from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)


def save_pdb(pdb_str, pdb_name, path: Path):
    with open(path.joinpath(pdb_name + ".pdb"), "w") as f:
        f.write(pdb_str)


def open_pdb(pdb_file):
    with open(pdb_file, "r") as f:
        pdb_str = f.read()
        return pdb_str
