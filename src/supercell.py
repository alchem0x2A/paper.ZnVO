import ase
import json
import numpy
import os, os.path
from ase.atom import Atom
from ase.parallel import parprint, world
from ase.build import add_adsorbate
from ase.build.supercells import make_supercell
from ase.constraints import FixAtoms
from gpaw import GPAW



# def gs_file(base_dir="./"):
    # assert name in ("ZnV2O5", "CoV2O5")
    # old_traj_filename = os.path.join(base_dir,
                                     # "relax.old.traj")
    # gpw_file = os.path.join(base_dir, "gs_single.gpw")
    # assert os.path.exists(old_traj_filename)
    # if os.path.exists(gpw_file):
        # parprint("GS file calculated!")
    # else:
        # curr_dir = os.path.dirname(os.path.abspath(__file__))
        # param_file = os.path.join(curr_dir, "params.json")
        # params = json.load(open(param_file, "r"))  # On each rank
        # world.barrier()
        # calc = GPAW(**params["gs"],
                    # txt=os.path.join(base_dir, "gs.txt"))
        # atoms = ase.io.read(old_traj_filename)  # last one!
        # atoms.set_calculator(calc)
        # atoms.get_potential_energy()
        # calc.write(gpw_file, mode="all")
        # world.barrier()
    # return gpw_file

def make_super(base_dir="./"):
    traj_fin_filename = os.path.join(base_dir,
                                     "relaxed.traj")
    assert os.path.exists(traj_fin_filename)  # Should exists
    atoms = ase.io.read(traj_fin_filename)      # final image
    super_atoms = make_supercell(atoms,
                                 numpy.diag([2, 2, 1]))
    # Add fixation for All VO
    scaled_pos = super_atoms.get_scaled_positions()
    lim = 0.3
    cond = list(numpy.where((scaled_pos[:, -1] < lim) \
                       | (scaled_pos[:, -1] > 1 - lim)))[0]
    parprint(cond)
    constraints = FixAtoms(list(cond))
    super_atoms.set_constraint(constraints)
    return super_atoms

def add_adatom(atoms, atom="Na", charge=1,
               scaled_pos=(0, 0, 1 / 2)):
    a = Atom(atom, position=numpy.dot(scaled_pos, atoms.get_cell()),
             charge=charge)
    atoms.append(a)
    return
    
