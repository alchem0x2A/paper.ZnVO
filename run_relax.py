import sys
import os, os.path
# May need this for the path issue for gpaw-python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.structure import get_structure
from src.relax import relax
import shutil
from ase.parallel import paropen, parprint, world, rank, broadcast


# Name=Zn,  Co
def main(name,
         # root="../ZnVO/",
         root="/cluster/scratch/ttian/ZnVO",
         clean=False):
    if name not in ("Zn", "Co"):
        return False

    # Directory
    if rank == 0:
        base_dir = os.path.join(root, "{}V2O5".format(name))
        if clean:
            shutil.rmtree(base_dir, ignore_errors=True)
            
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
    world.barrier()
    if clean:
        return                  # on all ranks

    atoms = get_structure(name)
    parprint(atoms)
    base_dir = os.path.join(root, "{}V2O5".format(name))
    relax(atoms, name=name, base_dir=base_dir)
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Only 1 parameter is needed!")
    elif len(sys.argv) == 2:
        formula = sys.argv[1]
        main(formula)
    elif len(sys.argv) == 3:
        formula = sys.argv[1]
        if sys.argv[2] == "clean":
            main(formula, clean=True)
        else:
            main(formula)
    else:
        raise ValueError("Parameter ill defined!")