import numpy
import os
import shutil
import json
import time

from ase.parallel import parprint, broadcast, world, rank
from ase.io import read
import ase.db
from ase.optimize import QuasiNewton, BFGS
from ase.constraints import UnitCellFilter, StrainFilter, ExpCellFilter
from ase.io.trajectory import Trajectory
from gpaw import Mixer, MixerSum, MixerDif
from gpaw import GPAW, PW, FermiDirac
from src.supercell import make_super, add_adatom

from ase.visualize import view

model_dict = dict(A=[(0, 0, 1 / 2)],
                  B=[(1 / 2, 1 / 2, 1 / 2)],
                  C=[(0, 0, 1 / 2),
                     (1 / 2, 1 / 2, 1 / 2)])

# Relax the atoms by recursively unitcell-position relaxation
def relax(base_dir="./",
          model="A",
          smax=2e-4, fmax=0.05):
    assert model.upper() in ("A", "B", "C")
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    param_file = os.path.join(curr_dir, "params.json")
    gpw_file = os.path.join(base_dir, "gs_{}.gpw".format(model.upper()))
    traj_filename = os.path.join(base_dir,
                                 "relax_{}.traj".format(model.upper()))
    traj_fin_filename = os.path.join(base_dir,
                                     "relax_fin_{}.traj".format(model.upper()))
    # Trajectory file from previous relaxation
    traj_base_filename = os.path.join(base_dir,
                                      "relaxed.traj")
    
    old_traj_filename = os.path.join(base_dir,
                                     "relax_{}.old.traj".format(model.upper()))
    log_filename = os.path.join(base_dir,
                                "relax_{}.log".format(model.upper()))
    if os.path.exists(gpw_file):
        parprint("Relaxation already done, will use gpw directly!")
        return 0
    if os.path.exists(param_file):
        params = json.load(open(param_file, "r"))
    else:
        raise FileNotFoundError("no parameter file!")
    # Perform relaxation process
    # If has trajectory, use the last image
    assert os.path.exists(traj_base_filename)

    # Get stress for original structure
    parprint("Calculate stress on original structure")
    calc = GPAW(**params["relax"],
                txt=os.path.join(base_dir, "gs.txt".format(model)))
    atoms_origin = Trajectory(traj_base_filename)[-1]
    atoms_origin.set_calculator(calc)
    s = atoms_origin.get_stress()
    parprint("Origin lattice, stress = ", s)

    del atoms_origin, calc
    
    # Now new calculations
    if os.path.exists(old_traj_filename):
        atoms = Trajectory(old_traj_filename)[-1]
    else:
        atoms = Trajectory(traj_base_filename)[-1]
        for pos in model_dict[model]:
            parprint("Add Na+ in position {}!".format(pos))
            add_adatom(atoms,
                       scaled_pos=pos)
    view(atoms)                 # Should be safe for batch systems

    # if os.path.exists(traj__filename):
        # parprint("Relaxed, now calculated GS!")
        # atoms = ase.io.read(traj_fin_filename)
        # calc = GPAW(**params["gs"],
                    # txt=os.path.join(base_dir, "gs.txt"))
        # atoms.set_calculator(calc)
        # atoms.get_potential_energy()
        # calc.write(gpw_file, mode="all")
        # return
    # else:
        # if os.path.exists(old_traj_filename):
            # try:
                # t = Trajectory(old_traj_filename)
                # atoms = t[-1]           # use the last image
                # parprint("Has trajectory file, use directly!")
                # parprint("New atom position", atoms)
            # except Exception as e:
                # parprint("Trajectory file may be corrupted!")

    world.barrier()
    if model in ("A", "B"):
        charge = 1
    else:
        charge = 2
    calc = GPAW(**params["relax"],
                charge=charge,
                txt=os.path.join(base_dir, "relax_{}.txt".format(model)))
    parprint(calc.parameters)
    parprint("Init magmom", atoms.get_initial_magnetic_moments())
    atoms.set_calculator(calc)
    # optimizations
    max_iter = 5
    sf = StrainFilter(atoms)
    # sf = ExpCellFilter(atoms)
    # opt_unitcell = BFGS(sf,
                        # logfile=log_filename,
                        # trajectory=traj_filename)
    # opt_unitcell.run(fmax=fmax)
    # shutil.copyfile(traj_filename, old_traj_filename)
    # for _ in range(max_iter):
    for i in range(max_iter):
        opt_cell = BFGS(sf,
                        logfile=log_filename,
                        trajectory=traj_filename)
        opt_norm = BFGS(atoms,
                        logfile=log_filename,
                        trajectory=traj_filename)
        opt_norm.run(fmax=fmax)
        s = atoms.get_stress()
        parprint("Cycle {}, Addition stress {}".format(i, s))
        opt_cell.run(fmax=fmax)
        # shutil.copyfile(traj_filename, old_traj_filename)

    # opt_cell.run(fmax=fmax)
    atoms.write(traj_fin_filename)
    # calc.set(**params["gs"],
             # txt=os.path.join(base_dir,
                              # "gs.txt"))
    # atoms.get_potential_energy()
    # calc.write(gpw_file, mode="all")
    return True
