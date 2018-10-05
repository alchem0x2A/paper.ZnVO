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
    calc = Vasp(**params["relax"])
    atoms.set_calculator(calc)
    parprint("VASP", calc.atoms)
    # optimizations
    atoms.get_potential_energy()
    return True
