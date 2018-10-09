import numpy
import os
import os.path
import json
import time

from ase.parallel import parprint
from ase.io import read
import ase.db
from ase.optimize import QuasiNewton
from ase.constraints import UnitCellFilter, StrainFilter
from ase.io.trajectory import Trajectory
from ase.parallel import parprint

from ase.calculators.vasp import Vasp

ggau = {"Zn": 0,
        "Co": 3.3,
        "V": 3.5,
        "H": 0,
        "O": 0}

# Possible for newest version
ggau_luj = {"Zn": {"L": -1, "U": 0.0, "J":0.0},
        "Co": {"L": 2, "U": 3.3, "J":0.0},
        "V": {"L": 2, "U": 3.5, "J":0.0},
        "H": {"L": -1, "U": 0.0, "J":0.0},
        "O": {"L": -1, "U": 0.0, "J":0.0}
}

def split_dict(atoms):
    elems = atoms.get_chemical_symbols()
    new_dict = {k:ggau_luj[k] for k in ggau_luj if k in elems}
    return new_dict

# Deprecated
# def gen_keys(atoms):
    # elems = atoms.get_chemical_symbols()
    # used = {key:False for key in ggau}
    # ldaul = []
    # ldauu = []
    # ldauj = []
    # for s in elems:
        # if used[s] is not True:
            # u = ggau[s]
            # ldauu.append(u)
            # if u != 0:
                # ldaul.append(2)
                # ldauj.append(0.0)
            # else:
                # ldaul.append(-1)
                # ldauj.append(0.0)
    # return ldaul, ldauu, ldauu
                
# Relax the atoms by recursively unitcell-position relaxation
# Now support for VASP
def relax(atoms, name="", base_dir="./",
          smax=2e-4, fmax=0.05):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.join(curr_dir, "params.json")
    if os.path.exists(param_file) is not True:
        return False
    params = json.load(open(param_file, "r"))
    # Perform relaxation process
    # If has trajectory, use the last image
    # Compulsory for VASP to change to the working directory!
    os.chdir(base_dir)
    parprint(os.path.abspath(os.path.curdir))
    # VASP restart only possible for CONTCAR existence
    if not os.path.exists("CONTCAR"):
        params["relax"]["restart"] = False
    params["relax"]["ldau"] = True
    # ldaul, ldauu, ldauj = gen_keys(atoms)
    params["relax"]["ldau_luj"] = split_dict(atoms)
    calc = Vasp(**params["relax"])
    atoms.set_calculator(calc)
    
    # print(atoms.get_chemical_symbols())
    # parprint("VASP", calc.atoms)
    # calc.clean()
    # optimizations
    atoms.get_potential_energy()
    return True
