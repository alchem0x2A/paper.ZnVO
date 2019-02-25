import sys
import os, os.path
# May need this for the path issue for gpaw-python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.structure import get_structure
from src.supercell import make_super, add_adatom
from src.neb import neb, calc_img
import shutil
from ase.parallel import paropen, parprint, world, rank, broadcast
from ase.visualize import view


# Name=Zn,  Co
def main(name,
         imag="init",
         root="/cluster/scratch/ttian/ZnVO",
         clean=False):
    assert imag in ("init", "final")
    
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

    base_dir = os.path.join(root, "{}V2O5".format(name))
    if imag == "init":
        calc_img(base_dir=base_dir, scaled_pos=(0, 0, 1 / 2), index=imag)
    else:
        calc_img(base_dir=base_dir, scaled_pos=(1 / 2, 1 / 2, 1 / 2), index=imag)
    return 0

if __name__ == "__main__":
    assert len(sys.argv) == 3
    mater = sys.argv[1]
    imag = sys.argv[2]
    main(name=mater, imag=imag)
