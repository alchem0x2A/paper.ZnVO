import numpy
import os
import os.path
import json
import time

from ase.parallel import parprint
from ase.io import read
import ase.db
from ase.optimize import QuasiNewton, BFGS
from ase.constraints import UnitCellFilter, StrainFilter
from ase.io.trajectory import Trajectory

from gpaw import GPAW, PW, FermiDirac


# Relax the atoms by recursively unitcell-position relaxation
def relax(atoms, name="", base_dir="./",
          smax=2e-4, fmax=0.05):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.join(curr_dir, "params.json")
    gpw_file = os.path.join(base_dir, "gs.gpw")
    if os.path.exists(gpw_file):
        parprint("Relaxation already done, will use gpw directly!")
        return 0
    if os.path.exists(param_file):
        params = json.load(open(param_file, "r"))
    else:
        raise FileNotFoundError("no parameter file!")
    # Perform relaxation process
    traj_filename = os.path.join(base_dir,
                                 "relax.traj")
    log_filename = os.path.join(base_dir,
                                "relax.log")
    # If has trajectory, use the last image
    if os.path.exists(traj_filename):
        parprint("Has rejectory file, use directly!")
        t = Trajectory(traj_filename)
        atoms = t[-1]           # use the last image
        parprint("New atom position", atoms)

    calc = GPAW(**params["relax"])
    atoms.set_calculator(calc)
    # optimizations
    max_iter = 3
    i = 0
        # unitcell optimization
    sf = StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0])
        # parprint(atoms, atoms.constraints)
        # atoms.set_constraint(sf)
        # parprint(atoms, atoms.constraints)
    opt_cell = BFGS(sf, logfile=log_filename,
                               trajectory=traj_filename)
        
    opt_cell.run(fmax=fmax)
    # parprint("Iter {}, unit cell optimization: stress {}, force {}".format(i,
                                                                           # max(atoms.get_stress()),
                                                                           # max(atoms.get_forces())))
        # remove the constraints
    atoms.set_constraint()  # Free!
    opt_norm = BFGS(atoms,
                    logfile=log_filename,
                    trajectory=traj_filename)
    opt_norm.run(fmax=fmax)
    # parprint("Iter {}, atoms optimization: stress {}, force {}".format(i,
                                                                           # max(atoms.get_stress()),
                                                                           # max(atoms.get_forces())))

    # If relaxation finished, perform ground state calculation
    calc.set(**params["gs"])
    atoms.get_potential_energy()
    calc.write(gpw_file, mode="all")
