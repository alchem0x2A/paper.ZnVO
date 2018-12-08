import numpy
import os
import os.path
import json
import time

from ase.parallel import parprint, broadcast, world, rank
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
        try:
            t = Trajectory(traj_filename)
            atoms = t[-1]           # use the last image
            parprint("Has rejectory file, use directly!")
            parprint("New atom position", atoms)
        except Exception as e:
            parprint("Trajectory file may be corrupted!")

    world.barrier()
    calc = GPAW(**params["relax"],
                txt=os.path.join(base_dir, "relax.txt"))
    parprint("Init magmom", atoms.get_initial_magnetic_moments())
    atoms.set_calculator(calc)
    # optimizations
    max_iter = 3
    sf = StrainFilter(atoms)
    for _ in range(max_iter):
        opt_cell = BFGS(sf,
                        logfile=log_filename,
                        trajectory=traj_filename)
        opt_norm = BFGS(atoms,
                        logfile=log_filename,
                        trajectory=traj_filename)
        opt_cell.run(fmax=fmax)
        opt_norm.run(fmax=fmax)
    atoms.write(os.path.join(base_dir, "relaxed.traj"))
    calc.set(**params["gs"])
    atoms.get_potential_energy()
    calc.write(gpw_file, mode="all")
    return True
