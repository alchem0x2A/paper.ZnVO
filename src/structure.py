import ase
from ase.io import read, write
from ase.atom import Atom
from ase.atoms import Atoms
import numpy
from numpy.linalg import norm
from numpy import sin, cos, arcsin, arccos, arctan
import math
from scipy.constants import pi, e
import sys
import os, os.path

# Only 1 vector now
def cart_to_sph(p):
    r = numpy.sqrt(numpy.sum(p ** 2))
    theta = arccos(p[2] / r)
    phi = arctan(p[1] / p[0])
    return r, theta, phi
    
def get_structure(name):
    angle_w = math.radians(108)
    l_OH = 0.96
    p1 = numpy.array([sin(angle_w / 2), cos(angle_w / 2), 0])
    p2 = numpy.array([-sin(angle_w / 2), cos(angle_w / 2), 0])
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    if "Zn" in name:
        f_name = os.path.join(curr_dir, "../data/Zn0.25V2O5(H2O)ICSD82328.cif")
    elif "Co" in name:
        f_name = os.path.join(curr_dir, "../data/Co0.25V2O5(H2O)ICSD50659.cif")
    else:
        return None
    mol = read(f_name)
    scaled_pos = mol.get_scaled_positions()
    ow = [(atom, i) for i, atom in enumerate(mol)\
          if (scaled_pos[i][-1] < 0.52) \
          and (scaled_pos[i][-1] > 0.48) \
          and (atom.symbol == "O")]
    metals = [atom for atom in mol if atom.symbol in ("Co", "Zn")]  # center of the octahedral
    # expand metals using unit cell
    cell = mol.cell
    m_pos = []
    for m in metals:
        for i in (-1, 0, 1):
            for j in range(3):
                m_pos.append(m.position + i * cell[j, :])  # toward direction j with repeat i
                # search for closet
    # Look for the smallest distance and place H in the plane
    for o, i in ow:
        dist = [norm(o.position - mp) for mp in m_pos]
        mp = m_pos[numpy.argmin(dist)]
        # print("Zn", mp, "O", o.position)
        vec = (o.position - mp) / norm(o.position - mp)  # unit vector from M->O
        r, theta, phi = cart_to_sph(vec)
        # print(vec, r, theta, phi)
        vec_xy = numpy.array([r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), 0])
        vec_n = numpy.cross(vec_xy, vec)
        vec_n = vec_n / numpy.linalg.norm(vec_n)
        
        # print(vec, p1)
        for p in (p1, p2):
            vec_ph = p[0] * vec_n + p[1] * vec  # reconstruct in the vec, vec_n plane
            ph = o.position + vec_ph * l_OH
            h = Atom("H", ph)
            mol.append(h)

    # Add initial magmom
    symbols = numpy.array(mol.get_chemical_symbols())  # for comparison
    magmom = numpy.zeros(len(symbols))
    magmom[symbols == "V"] = 0.25
    magmom[symbols == "Co"] = 2.0
    mol.set_initial_magnetic_moments(magmom)
    return mol

if __name__ == "__main__":
    from ase.visualize import view
    name = "Zn"
    mol = get_structure(name)
    view(mol)
