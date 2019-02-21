import ase
from ase.parallel import parprint, world
from ase.build import add_adsorbate
from gpaw import GPAW
import json
import numpy
import os, os.path


def gs_file(base_dir="./"):
    # assert name in ("ZnV2O5", "CoV2O5")
    old_traj_filename = os.path.join(base_dir,
                                     "relax.old.traj")
    gpw_file = os.path.join(base_dir, "gs_single.gpw")
    assert os.path.exists(old_traj_filename)
    if os.path.exists(gpw_file):
        parprint("GS file calculated!")
    else:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        param_file = os.path.join(curr_dir, "params.json")
        params = json.load(open(param_file, "r"))  # On each rank
        world.barrier()
        calc = GPAW(**params["gs"],
                    txt=os.path.join(base_dir, "gs.txt"))
        atoms = ase.io.read(old_traj_filename)  # last one!
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        calc.write(gpw_file, mode="all")
        world.barrier()
    return gpw_file


def create()
